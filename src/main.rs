use std::process::exit;

use config::Configuration;
use sw8s_rust_lib::comms::control_board::ControlBoard;
use tokio::{
    signal,
    sync::{
        mpsc::{self, UnboundedSender},
        OnceCell,
    },
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

#[tokio::main]
async fn main() {
    let shutdown_tx = shutdown_handler().await;
    let mut config = Configuration::default();

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
