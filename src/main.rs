use anyhow::{bail, Result};
use config::Configuration;
use std::path::Path;

use std::env;
use std::process::exit;
use sw8s_rust_lib::{
    comms::{
        control_board::{ControlBoard, SensorStatuses},
        meb::MainElectronicsBoard,
    },
    missions::{
        action::ActionExec,
        action_context::FullActionContext,
        basic::descend_and_go_forward,
        circle_buoy::{buoy_circle_sequence, buoy_circle_sequence_model},
        example::initial_descent,
        gate::{gate_run_complex, gate_run_naive, gate_run_testing},
        octagon::look_up_octagon,
        vision::PIPELINE_KILL,
    },
    video_source::appsink::Camera,
    vision::buoy::Target,
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

static CONTROL_BOARD_CELL: OnceCell<ControlBoard<WriteHalf<SerialStream>>> = OnceCell::const_new();
async fn control_board() -> &'static ControlBoard<WriteHalf<SerialStream>> {
    CONTROL_BOARD_CELL
        .get_or_init(|| async {
            ControlBoard::serial(&Configuration::default().control_board_path)
                .await
                .unwrap()
        })
        .await
}

static MEB_CELL: OnceCell<MainElectronicsBoard> = OnceCell::const_new();
async fn meb() -> &'static MainElectronicsBoard {
    MEB_CELL
        .get_or_init(|| async {
            MainElectronicsBoard::serial(&Configuration::default().meb_path)
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
                &Configuration::default().front_cam,
                "front",
                Path::new("/tmp/front_feed.mp4"),
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
                &Configuration::default().bottom_cam,
                "bottom",
                Path::new("/tmp/bottom_feed.mp4"),
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
    let _config = Configuration::default();

    for arg in env::args().skip(1).collect::<Vec<String>>() {
        run_mission(&arg).await.unwrap();
    }

    // Send shutdown signal
    shutdown_tx.send(()).unwrap();
}

/// Graceful shutdown, see <https://tokio.rs/tokio/topics/shutdown>
async fn shutdown_handler() -> UnboundedSender<()> {
    let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel::<()>();
    tokio::spawn(async move {
        // Wait for shutdown signal
        let exit_status =
            tokio::select! {_ = signal::ctrl_c() => { 1 }, _ = shutdown_rx.recv() => { 0 }};

        let status = control_board().await.sensor_status_query().await;

        match status.unwrap() {
            SensorStatuses::ImuNr => {
                println!("imu not ready");
                exit(1);
            }
            SensorStatuses::DepthNr => {
                println!("depth not ready");
                exit(1);
            }
            _ => {
                println!("all good");
            }
        }

        // Stop motors
        if let Some(control_board) = CONTROL_BOARD_CELL.get() {
            control_board
                .relative_dof_speed_set_batch(&[0.0; 6])
                .await
                .unwrap();
        };

        // If shutdown is unexpected, immediately exit nonzero
        if exit_status != 0 {
            exit(exit_status)
        };
    });
    shutdown_tx
}

async fn run_mission(mission: &str) -> Result<()> {
    let res = match mission.to_lowercase().as_str() {
        "arm" => {
            let cntrl_ready = tokio::spawn(async { control_board().await });
            println!("Waiting on MEB");
            while meb().await.thruster_arm().await != Some(true) {
                sleep(Duration::from_millis(10)).await;
            }
            println!("Got MEB");
            let _ = cntrl_ready.await;
            Ok(())
        }
        "empty" => {
            let control_board = control_board().await;
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                .await
                .unwrap();
            sleep(Duration::from_millis(1000)).await;
            println!("1");
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .await
                .unwrap();
            sleep(Duration::from_millis(1000)).await;
            println!("2");
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .await
                .unwrap();
            sleep(Duration::from_millis(1000)).await;
            println!("3");
            control_board
                .raw_speed_set([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .await
                .unwrap();
            println!("4");
            Ok(())
        }
        "depth_test" | "depth-test" => {
            let _control_board = control_board().await;
            println!("Init ctrl");
            sleep(Duration::from_millis(1000)).await;
            println!("End sleep");
            println!("Starting depth hold...");
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
            println!("Finished depth hold");
            Ok(())
        }
        "travel_test" | "travel-test" => {
            println!("Starting travel...");
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
            println!("Finished travel");
            Ok(())
        }
        "surface_" | "surface-test" => {
            println!("Starting travel...");
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
            println!("Finished travel");
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
            println!("Opening camera");
            front_cam().await;
            bottom_cam().await;
            println!("Opened camera");
            Ok(())
        }
        /*
        "path_align" => {
            bail!("TODO");
            let _ = path_align(&FullActionContext::new(
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
        "buoy_circle" => {
            bail!("TODO");
            let _ = gate_run(&FullActionContext::new(
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
        "look_up_octagon" => {
            let _ = look_up_octagon(&FullActionContext::new(
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
