use std::{env, process::exit, time::Duration};

use anyhow::{bail, Result};
use config::Configuration;
use sw8s_rust_lib::comms::{control_board::ControlBoard, meb::MainElectronicsBoard};
use tokio::{
    signal,
    sync::{
        mpsc::{self, UnboundedSender},
        OnceCell,
    },
    time::sleep,
};
use tokio_serial::SerialStream;

mod config;

static CONTROL_BOARD_CELL: OnceCell<ControlBoard<SerialStream>> = OnceCell::const_new();
async fn control_board() -> &'static ControlBoard<SerialStream> {
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

/// Graceful shutdown, see https://tokio.rs/tokio/topics/shutdown
async fn shutdown_handler() -> UnboundedSender<()> {
    let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel::<()>();
    tokio::spawn(async move {
        // Wait for shutdown signal
        let exit_status =
            tokio::select! {_ = signal::ctrl_c() => { 1 }, _ = shutdown_rx.recv() => { 0 }};

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
    match mission.to_lowercase().as_str() {
        "arm" => {
            let cntrl_ready = tokio::spawn(async { control_board().await });
            while meb().await.thruster_arm().await != Some(true) {
                sleep(Duration::from_millis(10)).await;
            }
            let _ = cntrl_ready.await;
            Ok(())
        }
        "depth_test" | "depth-test" => {
            println!("Starting depth hold...");
            control_board()
                .await
                .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
                .await?;
            sleep(Duration::from_secs(15)).await;
            println!("Finished depth hold");
            Ok(())
        }
        "travel_test" | "travel-test" => {
            println!("Starting travel...");
            control_board()
                .await
                .stability_2_speed_set(0.0, 0.5, 30.0, 30.0, 30.0, -1.0)
                .await?;
            sleep(Duration::from_secs(15)).await;
            println!("Finished travel");
            Ok(())
        }
        x => bail!("Invalid argument: [{x}]"),
    }
}
