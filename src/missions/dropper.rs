use tokio::io::WriteHalf;
use tokio_serial::SerialStream;
use std::time::Duration;
use crate::missions;
use missions::action_context::GetBottomCamMat;
use crate::missions::path_align::path_align;
use std::default::Default;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            TupleSecond,
        },
        basic::DelayAction,
        comms::StartBno055,
        extra::{AlwaysTrue, CountFalse, CountTrue, IsSome, OutputType, Terminal},
        fire_torpedo::{FireLeftTorpedo, FireRightTorpedo},
        movement::{
            AdjustType, ClampX, ConstYaw, LinearYawFromX, MultiplyX, OffsetToPose, ReplaceX, SetX,
            SetY, Stability2Adjust, Stability2Movement, Stability2Pos, ZeroMovement,
        },
        vision::{
            DetectTarget, ExtractPosition, MidPoint, Norm, SizeUnder, Vision, VisionSizeLock,
        },
    },
    vision::{
        buoy_model::{BuoyModel, Target},
        nn_cv2::OnnxModel,
        Offset2D,
    },
    POOL_YAW_SIGN,
};

use super::{action::ActionExec, action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard}};

// Constants for alignment and movement
const ALIGN_X_SPEED: f32 = 0.5; // Speed for lateral adjustment
const ALIGN_Y_SPEED: f32 = 0.5; // Speed for forward movement
const ALIGN_YAW_SPEED: f32 = 7.0; // Yaw adjustment speed
const DEPTH: f32 = 1.5; // Target depth for the robot
const PATH_DETECTION_THRESHOLD: f32 = 0.1; // Threshold for path detection

pub fn dropper<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin
        + GetBottomCamMat,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    act_nest!(
        ActionSequence::new,
        // Step 1: Initialize the BNO055 sensor
        StartBno055::new(context),

        // Step 2: Align with the path
        path_align(context),

        /**
        // Step 3: Follow the path
        act_nest!(
            ActionSequence::new,
            // Detect and extract the path position
            IsSome::new(ExtractPosition::new()), // Loop while the path exists
            act_nest!(
                ActionChain::new,
                OffsetToPose::new(Norm::default())), // Convert offset to movement command
                LinearYawFromX::new(ALIGN_YAW_SPEED), // Adjust yaw based on lateral offset
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(ALIGN_X_SPEED, ALIGN_Y_SPEED, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            OutputType::<()>::new(),
        ),
        */

        // Step 4: Perform the drop
        act_nest!(
            ActionSequence::new,
            // Stop the robot
            ZeroMovement::new(context, DEPTH),
            // Perform the drop (e.g., release an object)
            FireLeftTorpedo::new(context), // Replace with your drop mechanism
            OutputType::<()>::new(),
        )
    )
}