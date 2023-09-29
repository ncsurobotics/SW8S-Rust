use config::configuration;
use sw8s_rust_lib::comms::control_board::ControlBoard;
use tokio::sync::OnceCell;
use tokio_serial::SerialStream;

mod config;

static CONTROL_BOARD_CELL: OnceCell<ControlBoard<SerialStream>> = OnceCell::const_new();
async fn control_board() -> &'static ControlBoard<SerialStream> {
    CONTROL_BOARD_CELL
        .get_or_init(|| async {
            ControlBoard::serial(&configuration().read().unwrap().control_board_path)
                .await
                .unwrap()
        })
        .await
}

#[tokio::main]
async fn main() {
    println!("Hello, world!");
}
