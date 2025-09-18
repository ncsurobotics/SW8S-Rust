use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

use crate::config::path_align::Config;
use crate::config::ColorProfile;
use crate::{missions::vision::VisionNormBottomAngle, vision::path_cv::PathCV};

use super::{
    action::ActionExec,
    action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard},
};

pub async fn path_align_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
    config: &Config,
    color_profile: &ColorProfile,
) {
    #[cfg(feature = "logging")]
    logln!("Starting path align");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;
    let mut vision_norm_bottom = VisionNormBottomAngle::<Con, PathCV, f64>::new(
        context,
        PathCV::from_color_profile(color_profile),
    );

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    // let _ = cb
    //     .stability_1_speed_set(config.speed, 0.1, 0.0, 0.0, 0.0, config.depth)
    //     .await;
    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    sleep(Duration::from_secs(3)).await;

    let _ = cb
        .stability_2_speed_set(
            config.speed,
            config.forward_speed,
            0.0,
            0.0,
            initial_yaw,
            config.depth,
        )
        .await;

    let mut last_set_yaw = initial_yaw;
    let mut consec_detections = 0;

    #[cfg(feature = "logging")]
    logln!("Starting path detection");

    loop {
        if consec_detections >= config.detections {
            #[cfg(feature = "logging")]
            logln!("Finished path align");

            break;
        }

        if let Some(current_angle) = cb.responses().get_angles().await {
            let current_yaw = *current_angle.yaw();

            // For the opencv impl of path detection, the returned vector is guaranteed to contain 1 item
            #[allow(unused_variables)]
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
                y = -(*position.y() as f32);
                yaw = current_yaw + (*position.angle() * -1.0) as f32;

                last_set_yaw = yaw;
                consec_detections += 1;
            } else {
                consec_detections = 0;
                continue;
            }

            #[allow(unused_variables)]
            if let Err(e) = cb
                .stability_2_speed_set(x, y, 0.0, 0.0, last_set_yaw, config.depth)
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
    let _ = cb
        .stability_2_speed_set(0.0, 1.0, 0.0, 0.0, last_set_yaw, config.depth)
        .await;
    sleep(Duration::from_secs(1)).await;
}

pub async fn static_align_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting static align");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let target_yaw = initial_yaw + config.yaw_angle;

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    sleep(Duration::from_secs(config.yaw_wait)).await;

    let _ = cb
        .stability_2_speed_set(
            config.strafe_speed,
            config.forward_speed,
            0.0,
            0.0,
            target_yaw,
            config.depth,
        )
        .await;

    sleep(Duration::from_secs(config.forward_duration)).await;

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, target_yaw, config.depth)
        .await;
}
