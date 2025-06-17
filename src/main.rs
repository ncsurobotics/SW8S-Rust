use anyhow::{bail, Result};
use crossbeam::epoch::Pointable;
use std::env::temp_dir;

use std::env;
use std::process::exit;
use sw8s_rust_lib::{
    comms::{
        control_board::{ControlBoard, SensorStatuses},
        meb::MainElectronicsBoard,
    },
    config::Config,
    logln,
    missions::{
        action::ActionExec,
        action_context::FullActionContext,
        align_buoy::{buoy_align, buoy_align_shot},
        basic::descend_and_go_forward,
        circle_buoy::{
            buoy_circle_sequence, buoy_circle_sequence_blind, buoy_circle_sequence_model,
        },
        coinflip::coinflip,
        example::{initial_descent, pid_test},
        fancy_octagon::fancy_octagon,
        fire_torpedo::{FireLeftTorpedo, FireRightTorpedo},
        gate::{
            gate_run_coinflip, gate_run_complex, gate_run_naive, gate_run_procedural,
            gate_run_testing,
        },
        meb::WaitArm,
        octagon::octagon,
        path_align::path_align_procedural,
        reset_torpedo::ResetTorpedo,
        slalom::slalom,
        spin::spin,
        vision::PIPELINE_KILL,
    },
    video_source::appsink::Camera,
    vision::buoy::Target,
    TIMESTAMP,
};
use tokio::{
    io::WriteHalf,
    signal,
    sync::{
        mpsc::{self, UnboundedSender},
        OnceCell, RwLock,
    },
    time::{sleep, timeout},
};
use tokio_serial::SerialStream;
pub mod config;
use std::time::Duration;

static CONFIG_CELL: OnceCell<Config> = OnceCell::const_new();
async fn config() -> &'static Config {
    CONFIG_CELL
        .get_or_init(|| async {
            Config::new().unwrap_or_else(|e| {
                logln!("Error getting config file: {:#?}\nUsing default config", e);
                Config::default()
            })
        })
        .await
}

static CONTROL_BOARD_CELL: OnceCell<ControlBoard<WriteHalf<SerialStream>>> = OnceCell::const_new();
async fn control_board() -> &'static ControlBoard<WriteHalf<SerialStream>> {
    let config = config().await;
    CONTROL_BOARD_CELL
        .get_or_init(|| async {
            let board = ControlBoard::serial(config.control_board_path.as_str()).await;
            match board {
                Ok(x) => x,
                Err(e) => {
                    logln!("Error initializing control board: {:#?}", e);
                    let backup_board =
                        ControlBoard::serial(config.control_board_backup_path.as_str())
                            .await
                            .unwrap();
                    backup_board.reset().await.unwrap();
                    ControlBoard::serial(config.control_board_path.as_str())
                        .await
                        .unwrap()
                }
            }
        })
        .await
}

static MEB_CELL: OnceCell<MainElectronicsBoard<WriteHalf<SerialStream>>> = OnceCell::const_new();
async fn meb() -> &'static MainElectronicsBoard<WriteHalf<SerialStream>> {
    MEB_CELL
        .get_or_init(|| async {
            MainElectronicsBoard::<WriteHalf<SerialStream>>::serial(
                config().await.meb_path.as_str(),
            )
            .await
            .unwrap()
        })
        .await
}

static FRONT_CAM_CELL: OnceCell<Camera> = OnceCell::const_new();
async fn front_cam() -> &'static Camera {
    FRONT_CAM_CELL
        .get_or_init(|| async {
            Camera::jetson_new(
                config().await.front_cam_path.as_str(),
                "front",
                &temp_dir().join("cams_".to_string() + &TIMESTAMP),
            )
            .unwrap()
        })
        .await
}

static BOTTOM_CAM_CELL: OnceCell<Camera> = OnceCell::const_new();
async fn bottom_cam() -> &'static Camera {
    BOTTOM_CAM_CELL
        .get_or_init(|| async {
            Camera::jetson_new(
                config().await.bottom_cam_path.as_str(),
                "bottom",
                &temp_dir().join("cams_".to_string() + &TIMESTAMP),
            )
            .unwrap()
        })
        .await
}

static GATE_TARGET: OnceCell<RwLock<Target>> = OnceCell::const_new();
async fn gate_target() -> &'static RwLock<Target> {
    GATE_TARGET
        .get_or_init(|| async { RwLock::new(Target::Earth1) })
        .await
}

static STATIC_CONTEXT: OnceCell<FullActionContext<WriteHalf<SerialStream>>> = OnceCell::const_new();
async fn static_context() -> &'static FullActionContext<'static, WriteHalf<SerialStream>> {
    STATIC_CONTEXT
        .get_or_init(|| async {
            FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            )
        })
        .await
}

#[tokio::main]
async fn main() {
    let shutdown_tx = shutdown_handler().await;

    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        orig_hook(panic_info);
        exit(1);
    }));

    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        let meb = meb().await;

        // Wait for arm condition
        while meb.thruster_arm().await != Some(true) {
            sleep(Duration::from_secs(1)).await;
        }

        // Wait for disarm condition
        while meb.thruster_arm().await != Some(false) {
            sleep(Duration::from_secs(1)).await;
        }

        shutdown_tx_clone.send(1).unwrap();
    });

    for arg in env::args().skip(1).collect::<Vec<String>>() {
        run_mission(&arg).await.unwrap();
    }

    // Send shutdown signal
    shutdown_tx.send(0).unwrap();
}

/// Graceful shutdown, see <https://tokio.rs/tokio/topics/shutdown>
async fn shutdown_handler() -> UnboundedSender<i32> {
    let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel::<i32>();
    tokio::spawn(async move {
        // Wait for shutdown signal
        let exit_status = tokio::select! {_ = signal::ctrl_c() => {
        logln!("CTRL-C RECV");
        1 }, Some(x) = shutdown_rx.recv() => {
            logln!("SHUTDOWN SIGNAL RECV");
            x }};

        let status = control_board().await.sensor_status_query().await;

        match status.unwrap() {
            SensorStatuses::ImuNr => {
                logln!("imu not ready");
            }
            SensorStatuses::DepthNr => {
                logln!("depth not ready");
            }
            _ => {}
        }

        // Stop motors
        if let Some(control_board) = CONTROL_BOARD_CELL.get() {
            control_board
                .relative_dof_speed_set_batch(&[0.0; 6])
                .await
                .unwrap();
        };

        // Reset Torpedo
        ResetTorpedo::new(static_context().await).execute().await;

        // If shutdown is unexpected, immediately exit nonzero
        if exit_status != 0 {
            exit(exit_status)
        };
    });
    shutdown_tx
}

async fn run_mission(mission: &str) -> Result<()> {
    let config = config().await;
    let res = match mission.to_lowercase().as_str() {
        "arm" => {
            WaitArm::new(static_context().await).execute().await;
            Ok(())
        }
        "empty" => {
            let control_board = control_board().await;
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                .await
                .unwrap();
            sleep(Duration::from_millis(1000)).await;
            logln!("1");
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .await
                .unwrap();
            sleep(Duration::from_millis(1000)).await;
            logln!("2");
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .await
                .unwrap();
            sleep(Duration::from_millis(1000)).await;
            logln!("3");
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .await
                .unwrap();
            logln!("4");
            Ok(())
        }
        "depth_test" | "depth-test" => {
            let _control_board = control_board().await;
            logln!("Init ctrl");
            sleep(Duration::from_millis(1000)).await;
            logln!("End sleep");
            logln!("Starting depth hold...");
            loop {
                if let Ok(ret) = timeout(
                    Duration::from_secs(1),
                    control_board()
                        .await
                        .stability_1_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, -1.3),
                )
                .await
                {
                    ret?;
                    break;
                }
            }
            sleep(Duration::from_secs(5)).await;
            logln!("Finished depth hold");
            Ok(())
        }
        "travel_test" | "travel-test" => {
            logln!("Starting travel...");
            loop {
                if let Ok(ret) = timeout(
                    Duration::from_secs(1),
                    control_board()
                        .await
                        .stability_2_speed_set(0.0, 0.5, 0.0, 0.0, 70.0, -1.3),
                )
                .await
                {
                    ret?;
                    break;
                }
            }
            sleep(Duration::from_secs(10)).await;
            logln!("Finished travel");
            Ok(())
        }
        "surface_" | "surface-test" => {
            logln!("Starting travel...");
            loop {
                if let Ok(ret) = timeout(
                    Duration::from_secs(1),
                    control_board()
                        .await
                        .stability_1_speed_set(0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
                )
                .await
                {
                    ret?;
                    break;
                }
            }
            sleep(Duration::from_secs(10)).await;
            logln!("Finished travel");
            Ok(())
        }
        "descend" | "forward" => {
            let _ = descend_and_go_forward(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "gate_run_naive" => {
            let _ = gate_run_naive(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "gate_run_complex" => {
            let _ = gate_run_complex(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "gate_run_coinflip" => {
            // let _ = gate_run_coinflip(
            //     &FullActionContext::new(
            //         control_board().await,
            //         meb().await,
            //         front_cam().await,
            //         bottom_cam().await,
            //         gate_target().await,
            //     ),
            //     &config.missions.gate,
            // )
            // .execute()
            // .await;
            let _ = gate_run_procedural(
                &FullActionContext::new(
                    control_board().await,
                    meb().await,
                    front_cam().await,
                    bottom_cam().await,
                    gate_target().await,
                ),
                &config.missions.gate,
            )
            .await;

            Ok(())
        }
        "gate_run_testing" => {
            let _ = gate_run_testing(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "start_cam" => {
            // This has not been tested
            logln!("Opening camera");
            front_cam().await;
            bottom_cam().await;
            logln!("Opened camera");
            Ok(())
        }
        "path_align" => {
            let _ = path_align_procedural(
                &FullActionContext::new(
                    control_board().await,
                    meb().await,
                    front_cam().await,
                    bottom_cam().await,
                    gate_target().await,
                ),
                &config.missions.path_align,
            )
            .await;
            Ok(())
        }
        /*
        "buoy_circle" => {
            bail!("TODO");
            let _ = gate_run(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        */
        "example" => {
            let _ = initial_descent(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "pid_test" => {
            let _ = pid_test(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "octagon" => {
            let _ = octagon(static_context().await).execute().await;
            Ok(())
        }
        "fancy_octagon" => {
            let _ = fancy_octagon(static_context().await).execute().await;
            Ok(())
        }
        "buoy_circle" => {
            let _ = buoy_circle_sequence(&FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
                gate_target().await,
            ))
            .execute()
            .await;
            Ok(())
        }
        "buoy_model" => {
            let _ = buoy_circle_sequence_model(static_context().await)
                .execute()
                .await;
            Ok(())
        }
        "buoy_blind" => {
            let _ = buoy_circle_sequence_blind(static_context().await)
                .execute()
                .await;
            Ok(())
        }
        "buoy_align" => {
            let _ = buoy_align(static_context().await).execute().await;
            Ok(())
        }
        "spin" => {
            let _ = spin(static_context().await).execute().await;
            Ok(())
        }
        "torpedo" | "fire_torpedo" => {
            let _ = buoy_align_shot(static_context().await).execute().await;
            Ok(())
        }
        "torpedo_only" => {
            FireRightTorpedo::new(static_context().await)
                .execute()
                .await;
            FireLeftTorpedo::new(static_context().await).execute().await;
            Ok(())
        }
        "coinflip" => {
            let _ = coinflip(static_context().await).execute().await;
            Ok(())
        }
        // Just stall out forever
        "forever" | "infinite" => loop {
            while control_board().await.raw_speed_set([0.0; 8]).await.is_err() {}
            sleep(Duration::from_secs(u64::MAX)).await;
        },
        "open_cam_test" => {
            Camera::jetson_new(
                config.bottom_cam_path.as_str(),
                "front",
                &temp_dir().join("cams_".to_string() + &TIMESTAMP),
            )
            .unwrap();
            Ok(())
        }
        "slalom" => {
            let _ = slalom(
                &FullActionContext::new(
                    control_board().await,
                    meb().await,
                    front_cam().await,
                    bottom_cam().await,
                    gate_target().await,
                ),
                &config.missions.slalom,
            )
            .await;
            Ok(())
        }
        x => bail!("Invalid argument: [{x}]"),
    };

    // Kill any vision pipelines
    PIPELINE_KILL.write().unwrap().1 = true;
    while PIPELINE_KILL.read().unwrap().0 > 0 {
        sleep(Duration::from_millis(100)).await;
    }
    PIPELINE_KILL.write().unwrap().1 = false;

    res
}
