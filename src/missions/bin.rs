use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use super::action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard};

pub async fn bin<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
) {
    #[cfg(feature = "logging")]
    logln!("Starting bin");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;
    #[cfg(feature = "logging")]
    logln!("Finished bin");
}
