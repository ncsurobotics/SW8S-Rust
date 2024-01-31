use crate::vision::{buoy::Buoy, nn_cv2::OnnxModel, VisualDetector};

use super::{
    action::{Action, ActionExec, ActionSequence, ActionWhile},
    action_context::{GetControlBoard, GetFrontCamMat},
    basic::DelayAction,
    movement::{StraightMovement, ZeroMovement},
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;

use tokio_serial::SerialStream;

/// Action to drive to a Buoy using vision
/// will not set the power to zero on its own.
#[derive(Debug)]
pub struct DriveToBuoyVision<'a, T> {
    context: &'a T,
    target_depth: f32,
    forward_power: f32,
    k_p: f32,
    buoy_model: Buoy<OnnxModel>,
}

impl<'a, T> DriveToBuoyVision<'a, T> {
    pub fn new(context: &'a T, target_depth: f32, forward_power: f32) -> Self {
        DriveToBuoyVision {
            context,
            target_depth,
            forward_power,
            k_p: 0.3,
            buoy_model: Buoy::default(),
        }
    }
}

impl<T> Action for DriveToBuoyVision<'_, T> {}

#[async_trait]
impl<T> ActionExec for DriveToBuoyVision<'_, T>
where
    T: GetControlBoard<WriteHalf<SerialStream>> + GetFrontCamMat + Sync + Unpin,
{
    type Output = Result<()>;

    async fn execute(&mut self) -> Self::Output {
        let camera_aquisition = self.context.get_front_camera_mat();
        let class_of_interest = self.context.get_desired_buoy_gate().await;

        let model_acquisition = self.buoy_model.detect(&camera_aquisition.await);
        match model_acquisition {
            Ok(acquisition_vec) if !acquisition_vec.is_empty() => {
                let detected_item = acquisition_vec
                    .iter()
                    .find(|&result| *result.class() == class_of_interest);

                if let Some(scan) = detected_item {
                    let position = self.buoy_model.normalize(scan.position());
                    self.context
                        .get_control_board()
                        .stability_2_speed_set_initial_yaw(
                            self.forward_power,
                            self.k_p * position.x as f32,
                            0.0,
                            0.0,
                            self.target_depth,
                        )
                        .await?;
                    Ok(()) // Repeat the action
                } else {
                    Err(anyhow::format_err!("no longer detected")) // Stop the action
                }
            }
            _ => {
                Err(anyhow::format_err!(
                    "No buoy detected or error in detection"
                )) // Stop the action
            }
        }
    }
}

pub fn buoy_bump_sequence<'a, T>(
    context: &'a T,
    depth: f32,
) -> ActionSequence<
    ActionWhile<DriveToBuoyVision<'a, T>>,
    ActionSequence<StraightMovement<T>, ActionSequence<DelayAction, ZeroMovement<'a, T>>>,
>
where
    T: GetControlBoard<WriteHalf<SerialStream>> + 'a,
{
    let forward_power = 0.3;
    let delay_s = 6.0;

    // Instantiate DriveToBuoyVision with provided values
    let drive_to_buoy_vision = DriveToBuoyVision::new(context, depth, forward_power);
    let drive_while_buoy_visible = ActionWhile::new(drive_to_buoy_vision);

    let forward_action = StraightMovement::new(context, depth, true);
    // Create a DelayAction with hardcoded delay
    let delay_action = DelayAction::new(delay_s);

    // Instantiate ZeroMovement with provided values
    let zero_movement = ZeroMovement::new(context, depth);

    // Create the inner ActionSequence
    let inner_sequence = ActionSequence::new(
        forward_action,
        ActionSequence::new(delay_action, zero_movement),
    );
    // Create and return the outer ActionSequence
    ActionSequence::new(drive_while_buoy_visible, inner_sequence)
}
