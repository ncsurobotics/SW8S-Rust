use itertools::Itertools;
use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

use super::action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::{
    config::slalom::Config,
    config::slalom::Side::*,
    missions::{action::ActionExec, vision::VisionNorm},
    vision::{nn_cv2::OnnxModel, slalom::Slalom, slalom::Target},
};

pub async fn slalom<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting slalom");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let mut vision = VisionNorm::<Con, Slalom<OnnxModel>, f64>::new(context, Slalom::default());

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let mut start_detections = 0;
    let mut end_detections = 0;

    // Begin stationary, we will only start forward if we detect nothing
    let _ = cb
        .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    loop {
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        if detections.len() == 0 {
            // We see nothing, meaning we have either not reached slalom yet, passed through it already, or are misaligned.
            // If we have not reached it yet, we need to go forward
            // If we have passed through already, end the mission
            if start_detections > config.start_detections {
                if end_detections > config.end_detections {
                    break;
                } else {
                    end_detections += 1;
                    #[cfg(feature = "logging")]
                    logln!("End count: {}/{}", end_detections, config.end_detections);
                }
            } else {
                #[cfg(feature = "logging")]
                logln!(
                    "Resetting start counter from {}/{}",
                    start_detections,
                    config.start_detections
                );
                start_detections = 0;
                let _ = cb
                    .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
                    .await;
            }
            continue;
        } else {
            if start_detections < config.start_detections {
                start_detections += 1;
                #[cfg(feature = "logging")]
                logln!(
                    "Start count: {}/{}",
                    start_detections,
                    config.start_detections
                );
            } else {
                #[cfg(feature = "logging")]
                logln!(
                    "Resetting end counter from {}/{}",
                    end_detections,
                    config.end_detections
                );
                end_detections = 0;
            }
        }

        let middle = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Middle))
            .collect_vec();

        let side = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Side))
            .collect_vec();

        if middle.len() > 0 {
            // Use the average x coord of all middle detections as the middle of slalom
            let middle_x =
                middle.iter().map(|d| *d.position().x() as f32).sum::<f32>() / middle.len() as f32;

            if side.len() > 0 {
                #[cfg(feature = "logging")]
                logln!("Got middle and side poles");
                // We have the middle and at least one side pole
                // Split the side pole detections into left and right groups based on their x coords
                let mut sum_l = 0.0;
                let mut sum_r = 0.0;
                let mut count_l = 0.0;
                let mut count_r = 0.0;

                for x in side.iter().map(|d| *d.position().x() as f32) {
                    // If the side poles x coord is greater than the middle pole, it's on the left
                    // Haven't checked if this is actually the right side, it might be flipped
                    if x > middle_x {
                        sum_l += x;
                        count_l += 1.0;
                    } else if x < middle_x {
                        sum_r += x;
                        count_r += 1.0;
                    }
                }

                let (sum, count) = match config.side {
                    Left => (sum_l, count_l),
                    Right => (sum_r, count_r),
                };

                // Find average x coords for l/r group of side poles
                let side_x = if count > 0.0 { sum / count } else { 0.0 };

                // Set target x to the average of the chosen side pole and middle pole's x coords
                let x = (side_x + middle_x) / 2.0;

                // Strafe to the left right towards the midpoint between the middle pole and the chosen side pole
                let _ = cb
                    .stability_2_speed_set(
                        x as f32,
                        config.speed,
                        0.0,
                        0.0,
                        initial_yaw,
                        config.depth,
                    )
                    .await;
            } else {
                // We have no side pole, so just align to the middle pole
                // Once the middle poles x is below a certain value, do a dumb translation in the correct direction
                if middle_x <= config.centered_threshold {
                    #[cfg(feature = "logging")]
                    logln!("Got only middle pole, finished centering");
                    let x = match config.side {
                        Left => config.speed,
                        Right => config.speed * -1.0,
                    };

                    let _ = cb
                        .stability_2_speed_set(x, config.speed, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                    sleep(Duration::from_secs(config.dumb_strafe_secs)).await;
                    let _ = cb
                        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                } else {
                    #[cfg(feature = "logging")]
                    logln!("Got only middle pole, centering");
                    let _ = cb
                        .stability_2_speed_set(middle_x, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                }
            }
        } else {
            if side.len() > 0 {
                #[cfg(feature = "logging")]
                logln!("Got only side poles");
                // We see at least a side pole, but no middle
                // This means we are probably seeing a side pole, so translate towards it
                let x =
                    side.iter().map(|d| *d.position().x() as f32).sum::<f32>() / side.len() as f32;
                let _ = cb
                    .stability_2_speed_set(x, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                    .await;
            }
        }
    }

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    #[cfg(feature = "logging")]
    logln!("Finished slalom");
}
