use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

use crate::{act_nest, missions::vision::VisionNormBottomAngle, vision::path_cv::PathCV};

use super::{
    action::ActionExec,
    action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard},
};

pub async fn path_align_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
) {
    const DEPTH: f32 = -1.25;
    const PATH_ALIGN_SPEED: f32 = 0.3;
    const DETECTIONS: u8 = 10;

    #[cfg(feature = "logging")]
    logln!("Starting path align");

    let cb = context.get_control_board();
    cb.bno055_periodic_read(true).await;
    let mut vision_norm_bottom =
        VisionNormBottomAngle::<Con, PathCV, f64>::new(context, PathCV::default());

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let _ = cb
        .stability_2_speed_set(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, initial_yaw, DEPTH)
        .await;

    let mut last_set_yaw = initial_yaw;
    let mut consec_detections = 0;

    #[cfg(feature = "logging")]
    logln!("Starting path detection");

    loop {
        if consec_detections >= DETECTIONS {
            #[cfg(feature = "logging")]
            logln!("Finished path align");

            break;
        }

        if let Some(current_angle) = cb.responses().get_angles().await {
            let current_yaw = *current_angle.yaw() as f32;

            // For the opencv impl of path detection, the returned vector is guaranteed to contain 1 item
            let detections = vision_norm_bottom.execute().await.unwrap_or_else(|e| {
                #[cfg(feature = "logging")]
                logln!(
                    "Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec"
                );
                vec![]
            });

            let mut positions = detections
                .into_iter()
                .filter_map(|d| d.class().then_some(d.position().clone()));

            let x;
            let y;
            let yaw;

            if let Some(position) = positions.next() {
                x = *position.x() as f32;
                y = (*position.y() as f32) * -1.0;
                yaw = current_yaw + (*position.angle() * -1.0) as f32;

                last_set_yaw = yaw;
                consec_detections += 1;
            } else {
                x = 0.0;
                y = PATH_ALIGN_SPEED;
                yaw = last_set_yaw;

                consec_detections = 0;
            }

            if let Err(e) = cb
                .stability_2_speed_set(x, y, 0.0, 0.0, last_set_yaw, DEPTH)
                .await
            {
                #[cfg(feature = "logging")]
                logln!("SASSIST2 command to cb resulted in error: `{e}`");
            }
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get current angle");
        }

        #[cfg(feature = "logging")]
        logln!("Positive detection count: {consec_detections}");
    }
    cb.stability_2_speed_set(0.0, 1.0, 0.0, 0.0, last_set_yaw, DEPTH)
        .await;
    sleep(Duration::from_secs(1)).await;
}
