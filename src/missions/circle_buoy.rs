use crate::vision::{buoy::{self, PCA}, nn_cv2::OnnxModel, VisualDetector};

use super::{
    action::{Action, ActionExec, ActionSequence, ActionWhile},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    movement::{StraightMovement, ZeroMovement},
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;


pub struct FindBuoy<'a, T> {
    context: &'a T,
    target_depth: f32,
    lateral_power: f32,

}

pub struct CircleBuoy <'a, T> {
    context: &'a T,
    target_depth: f32,
    lateral_power: f32,
}


impl<'a, T> FindBuoy<'a, T> {
    pub fn new(context: &'a T) -> Self {
        FindBuoy {
            context,
            target_depth,
            lateral_power: f32,
        }
    }
}

impl<'a, T> CircleBuoy <'a, T> {
    pub fn new(context: &'a T, target_depth: f32, forward_power: f32) -> Self {
        CircleBuoy {
            context,
            target_depth,
            lateral_power,
        }
    }
}

impl<T> Action for CircleBuoy<'_, T> {}
impl<T> Action for FindBuoy<'_, T> {}

#[async_trait]
impl<T> ActionExec for FindBuoy<'_, T>
where 
    T: GetControlBoard<WriteHalf<SerialStream>> + GetFrontCamMat + Sync + Unpin,
{
    async fn execute(&mut self) -> Self::Output {
        let camera_aquisition = self.context.get_front_camera_mat();

    }
}
#[async_trait]
impl<T> ActionExec for CircleBuoy<'_, T>
where
    T: GetControlBoard<WriteHalf<SerialStream>> + GetFrontCamMat + Sync + Unpin,
{
    async fn execute(&mut self) {
        self.context.get_control_board().stability_2_speed_set();
    }
}

pub fn buoy_circle_sequence<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &Con,
) -> impl ActionExec + '_ {
    const DEPTH: f32 = 1.0;

    let lateral_power = 0.3;
    let delay_s = 5.0;
    // Create a DelayAction with hardcoded delay
    let delay_action = DelayAction::new(delay_s);

    // Instantiate ZeroMovement with provided values
    let zero_movement = ZeroMovement::new(context, DEPTH);

    // Create the inner ActionSequence
    let inner_sequence = ActionSequence::new(
        ActionSequence::new(delay_action, zero_movement),
        ActionParallel::new(FindBuoy::new(), CircleBuoy::new())
    );
}
