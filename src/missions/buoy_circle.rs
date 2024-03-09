use crate::vision::{
    buoy::{self, Buoy},
    nn_cv2::OnnxModel,
    VisualDetector,
};

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

/*
pub fn buoy_circle<
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

    let forward_power = 0.3;
    let delay_s = 6.0;

    // Instantiate DriveToBuoyVision with provided values

    let drive_to_buoy_vision = DriveToBuoyVision::new(context, DEPTH, forward_power);
    let drive_while_buoy_visible = ActionWhile::new(drive_to_buoy_vision);

    let forward_action = StraightMovement::new(context, DEPTH, true);
    // Create a DelayAction with hardcoded delay
    let delay_action = DelayAction::new(delay_s);

    // Instantiate ZeroMovement with provided values
    let zero_movement = ZeroMovement::new(context, DEPTH);

    // Create the inner ActionSequence
    let inner_sequence = ActionSequence::new(
        forward_action,
        ActionSequence::new(delay_action, zero_movement),
    );
    // Create and return the outer ActionSequence
    ActionSequence::new(drive_while_buoy_visible, inner_sequence)
}
*/
