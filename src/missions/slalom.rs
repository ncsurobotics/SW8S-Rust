use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use super::action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::config::slalom::Config;

pub async fn slalom<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting slalom");

    let cb = context.get_control_board();
    cb.bno055_periodic_read(true).await;

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let _ = cb
        .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
        .await;
}
