use itertools::Itertools;
use tokio::io::WriteHalf;
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
    // There should be some sort of initial detections threshold to determine when we've actuallly reached slalom
    // After it has been reached, we *should* be getting continuous detections until we've cleared slalom
    // It's possible that we could get false positives for having cleard slalom if we get misaligned
    //   - This shouldn't really happen? If we always translate towards detected poles and maintain our initial yaw
    //     - Initial yaw is basically obtained from path, which should be pointed directly at slalom
    //
    // I currently have zero forward speed when trying to align, but it may be better to have a constant very slow forward speed
    #[cfg(feature = "logging")]
    logln!("Starting slalom");

    let cb = context.get_control_board();
    cb.bno055_periodic_read(true).await;

    let mut vision = VisionNorm::<Con, Slalom<OnnxModel>, f64>::new(context, Slalom::default());

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    // Begin stationary, we will only start forward if we detect nothing
    let _ = cb
        .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    #[cfg(feature = "logging")]
    logln!("Starting slalom");

    loop {
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        // Detecting nothing, go forward
        if detections.len() == 0 {
            let _ = cb
                .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
                .await;
            continue;
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
            let middle_x = middle.iter().map(|d| d.position().x()).sum() / middle.len();

            if side.len() > 0 {
                // We have the middle and at least one side pole
                // Split the side pole detections into left and right groups based on their x coords
                let mut sum_l = 0.0;
                let mut sum_r = 0.0;
                let mut count_l = 0;
                let mut count_r = 0;

                for x in side.iter().map(|d| d.position().x()) {
                    // If the side poles x coord is greater than the middle pole, it's on the left
                    // Haven't checked if this is actually the right side, it might be flipped
                    if x > middle_x {
                        sum_l += x;
                        count_l += 1;
                    } else if x < middle_x {
                        sum_r += x;
                        count_r += 1;
                    }
                }

                // Find average x coords for left and right groups of side poles
                let left_side_x = if count_l > 0 { sum_l / count_l } else { 0.0 };
                let right_side_x = if count_r > 0 { sum_r / count_r } else { 0.0 };

                // Set target x to the average of the chosen side pole and middle pole's x coords
                let x;
                match config.side {
                    Left => {
                        x = (left_side_x + middle_x) / 2;
                    }
                    Right => {
                        x = (right_side_x + middle_x) / 2;
                    }
                }
                // Strafe to the left right towards the midpoint between the middle pole and the chosen side pole
                let _ = cb
                    .stability_2_speed_set(x, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                    .await;
            } else {
                // We have no side pole, so just align to the middle pole
                // Once the middle poles x is below a certain value, do a dumb translation to correct direction
                if middle_x <= config.centered_threshold {
                    let _ = cb
                        .stability_2_speed_set(
                            0.0,
                            config.speed,
                            0.0,
                            0.0,
                            initial_yaw,
                            config.depth,
                        )
                        .await;
                } else {
                    let _ = cb
                        .stability_2_speed_set(middle_x, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                }
            }
        } else {
            if side.len() > 0 {
                // We see at least a side pole, but no middle
                // This probably means we should rotate towards the side pole to find the middle pole
                // Once we find it, go straight towards it, out yaw should get reset to the initial yaw later
                unimplemented!();
            } else {
                // We see nothing, meaning we have either not reached slalom yet, passed through it already, or are misaligned.
                // If we have not reached it yet, we need to go forward
                // If we have passed through already, end the mission (but how do you know? True/false counts)
                // If we are misaligned (aka neither of the above cases) we can just start spinning
                unimplemented!();
            }
        }
    }
}
