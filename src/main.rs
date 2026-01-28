use anyhow::{bail, Result};
use std::env::temp_dir;

use std::env;
use std::process::exit;
use std::time::Duration;
use sw8s_rust_lib::{
    comms::{
        control_board::{ControlBoard, SensorStatuses},
        meb::MainElectronicsBoard,
    },
    config::{Config, SHUTDOWN_TIMEOUT},
    logln,
    missions::{
        action::ActionExec,
        action_context::FullActionContext,
        basic::descend_and_go_forward,
        bin::bin,
        coinflip::coinflip_procedural,
        example::{initial_descent, pid_test},
        fire_torpedo::{FireLeftTorpedo, FireRightTorpedo},
        gate::{gate_run_cv_procedural, gate_run_dead_reckon, gate_run_procedural},
        meb::WaitArm,
        octagon::octagon,
        path_align::{path_align_procedural, static_align_procedural},
        reset_torpedo::ResetTorpedo,
        slalom::slalom,
        sonar::sonar,
        spin::spin,
        vision::PIPELINE_KILL,
    },
    video_source::appsink::Camera,
    TIMESTAMP,
};
use tokio::{
    io::WriteHalf,
    signal,
    sync::{
        mpsc::{self, UnboundedSender},
        OnceCell, Semaphore,
    },
    time::{sleep, timeout},
};
use tokio_serial::SerialStream;
use tokio_util::sync::CancellationToken;

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

static STATIC_CONTEXT: OnceCell<FullActionContext<WriteHalf<SerialStream>>> = OnceCell::const_new();
async fn static_context() -> &'static FullActionContext<'static, WriteHalf<SerialStream>> {
    STATIC_CONTEXT
        .get_or_init(|| async {
            FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
            )
        })
        .await
}

static SHUTDOWN_GUARD: Semaphore = Semaphore::const_new(1);

#[tokio::main]
async fn main() {
    let (shutdown_tx, mission_ct) = shutdown_handler().await;

    let orig_hook = std::panic::take_hook();
    let mission_ct_clone = mission_ct.clone();
    std::panic::set_hook(Box::new(move |panic_info| {
        orig_hook(panic_info);
        // Cancel running missions
        mission_ct_clone.cancel();
        // Wait for running mission to exit
        // handle.block_on(async {
        //     if let Err(_) = timeout(
        //         Duration::from_secs(SHUTDOWN_TIMEOUT),
        //         SHUTDOWN_GUARD.acquire(),
        //     )
        //     .await
        //     {
        //         logln!("Missions did not exit within {SHUTDOWN_TIMEOUT} seconds")
        //     }
        // });
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
        let _guard = SHUTDOWN_GUARD.acquire().await.unwrap();
        run_mission(&arg, mission_ct.clone()).await.unwrap();
    }

    // Send shutdown signal
    shutdown_tx.send(0).unwrap();
}

/// Graceful shutdown, see <https://tokio.rs/tokio/topics/shutdown>
async fn shutdown_handler() -> (UnboundedSender<i32>, CancellationToken) {
    let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel::<i32>();
    let mission_ct = CancellationToken::new();
    let mission_ct_clone = mission_ct.clone();
    tokio::spawn(async move {
        // Wait for shutdown signal
        let exit_status = tokio::select! {
            _ = signal::ctrl_c() => {
                logln!("CTRL-C RECV");
                1
            },
            Some(x) = shutdown_rx.recv() => {
                logln!("SHUTDOWN SIGNAL RECV");
                x
            }
        };

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

        // If shutdown is unexpected, cancel running missions and exit nonzero
        if exit_status != 0 {
            // Cancel running missions
            mission_ct_clone.cancel();
            // Wait for running mission to exit
            if timeout(
                Duration::from_secs(SHUTDOWN_TIMEOUT),
                SHUTDOWN_GUARD.acquire(),
            )
            .await
            .is_err()
            {
                logln!("Missions did not exit within {SHUTDOWN_TIMEOUT} seconds")
            }
            exit(exit_status)
        };
    });
    (shutdown_tx, mission_ct)
}

async fn run_mission(mission: &str, cancel: CancellationToken) -> Result<()> {
    /// Wrapper for missions that do not directly use the cancellation token
    macro_rules! ctwrap {
        ($fut:expr) => {{
            let _ = cancel.run_until_cancelled($fut).await;
            Ok(())
        }};
    }

    let config = config().await;
    let res = match mission.to_lowercase().as_str() {
        "arm" => ctwrap!(WaitArm::new(static_context().await).execute()),
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
        "descend" | "forward" => ctwrap!(descend_and_go_forward(&FullActionContext::new(
            control_board().await,
            meb().await,
            front_cam().await,
            bottom_cam().await,
        ))
        .execute()),
        "gate_run_coinflip" => ctwrap!(gate_run_cv_procedural(
            &FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
            ),
            &config.missions.gate,
            &config.get_color_profile().unwrap(),
        )),
        "gate_run_yolo" => ctwrap!(gate_run_procedural(
            &FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
            ),
            &config.missions.gate
        )),
        "gate_run_reckon" => ctwrap!(gate_run_dead_reckon(
            &FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
            ),
            &config.missions.gate,
        )),
        "start_cam" => {
            // This has not been tested
            logln!("Opening camera");
            front_cam().await;
            bottom_cam().await;
            logln!("Opened camera");
            Ok(())
        }
        "path_align" => ctwrap!(path_align_procedural(
            &FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
            ),
            &config.missions.path_align,
            &config.get_color_profile().unwrap(),
        )),
        "static_align" => ctwrap!(static_align_procedural(
            &FullActionContext::new(
                control_board().await,
                meb().await,
                front_cam().await,
                bottom_cam().await,
            ),
            &config.missions.path_align,
        )),
        "example" => ctwrap!(initial_descent(&FullActionContext::new(
            control_board().await,
            meb().await,
            front_cam().await,
            bottom_cam().await,
        ))
        .execute()),
        "pid_test" => ctwrap!(pid_test(&FullActionContext::new(
            control_board().await,
            meb().await,
            front_cam().await,
            bottom_cam().await,
        ))
        .execute()),
        "octagon" => ctwrap!(octagon(
            static_context().await,
            &config.missions.octagon,
            &config.get_color_profile().unwrap()
        )
        .execute()),
        "spin" => ctwrap!(spin(static_context().await, &config.missions.spin)),
        "torpedo_only" => {
            FireRightTorpedo::new(static_context().await)
                .execute()
                .await;
            FireLeftTorpedo::new(static_context().await).execute().await;
            Ok(())
        }
        "coinflip" => {
            ctwrap!(coinflip_procedural(
                static_context().await,
                &config.missions.coinflip
            ))
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
        "slalom_left" => ctwrap!(slalom(
            static_context().await,
            &config.missions.slalom,
            false,
            &config.get_color_profile().unwrap()
        )),
        "slalom_right" => ctwrap!(slalom(
            static_context().await,
            &config.missions.slalom,
            true,
            &config.get_color_profile().unwrap()
        )),
        "sonar" => {
            let _ = sonar(static_context().await, &config.sonar, cancel).await;
            Ok(())
        }
        "bin" => ctwrap!(bin(static_context().await)),
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
