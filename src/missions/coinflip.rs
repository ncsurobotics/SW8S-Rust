use itertools::Itertools;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    config::coinflip::Config,
    vision::{
        gate_poles::{GatePoles, Target},
        nn_cv2::OnnxModel,
    },
};

use super::{
    action::ActionExec,
    action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard},
    vision::VisionNorm,
};

pub async fn coinflip_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting path align");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;
    let mut vision =
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default());

    // let initial_yaw = loop {
    //     if let Some(initial_angle) = cb.responses().get_angles().await {
    //         break *initial_angle.yaw();
    //     } else {
    //         #[cfg(feature = "logging")]
    //         logln!("Failed to get initial angle");
    //     }
    // };

    let depth = config.depth;

    let _ = cb
        .stability_1_speed_set(0.0, 0.0, config.angle_correction, 0.0, 0.0, depth)
        .await;

    let mut true_count = 0;
    let max_true_count = config.true_count;

    loop {
        #[allow(unused_variables)]
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        // let gate = detections
        //     .iter()
        //     .filter(|d| matches!(d.class().identifier, Target::Gate))
        //     .collect_vec();
        let shark = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Shark))
            .collect_vec();
        let sawfish = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Sawfish))
            .collect_vec();

        let left_pole = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::LeftPole))
            .collect_vec();

        // let right_pole = detections
        //     .iter()
        //     .filter(|d| matches!(d.class().identifier, Target::RightPole))
        //     .collect_vec();

        if shark.len() > 0 || sawfish.len() > 0 || !left_pole.is_empty() {
            if true_count > max_true_count {
                let _ = cb
                    .stability_1_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, depth)
                    .await;
                break;
            } else {
                true_count += 1;
            }
        } else {
            true_count = 0;
        }
    }
}
