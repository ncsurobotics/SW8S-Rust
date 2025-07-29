use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use super::action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::{
    config::bin::Config,
    missions::{action::ActionExec, vision::VisionNormBottom},
    vision::{
        bin::Bin,
        nn_cv2::OnnxModel,
    },
};

pub async fn bin<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting bin");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let mut vision = VisionNormBottom::<Con, Bin<OnnxModel>, f64>::new(context, Bin::default());

    loop {
        #[cfg(feature = "logging")]
        logln!("DOING BIN DETECTION");
        if let Ok(detections) = vision.execute().await {
            unimplemented!();
        }
    }

    #[cfg(feature = "logging")]
    logln!("Finished bin");
}
