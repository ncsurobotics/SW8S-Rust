use tokio::{
    io::WriteHalf,
    select,
    time::{sleep, Duration},
};
use tokio_serial::{SerialPort, SerialPortBuilderExt, SerialStream};
use tokio_util::sync::CancellationToken;

use bluerobotics_ping::{
    device::{Ping360, PingDevice},
    ping360::AutoDeviceDataStruct,
};

use super::action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::{
    config::bin::Config,
    missions::{action::ActionExec, vision::VisionNormBottom},
    vision::{
        bin::{Bin, Target},
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
