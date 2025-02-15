use tokio::io::WriteHalf;
use tokio_serial::SerialStream;
use crate::missions::action_context::GetControlBoard;
use crate::missions::action_context::GetFrontCamMat;
use crate::missions::action_context::GetMainElectronicsBoard;
use crate::missions::action::ActionExec;

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

// Comments: Align_buoy is a good reference!

// Constants for alignment and movement
const ALIGN_X_SPEED: f32 = 0.5; // Speed for lateral adjustment
const ALIGN_Y_SPEED: f32 = 0.5; // Speed for forward movement
const ALIGN_YAW_SPEED: f32 = 7.0; // Yaw adjustment speed
const DEPTH: f32 = 1.5; // Target depth for the robot
const PATH_DETECTION_THRESHOLD: f32 = 0.1; // Threshold for path detection

pub fn dropper<
    // Uses a Con (Context) type with the traits listed
    // after the colon
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    // Needs a context given to it
    context: &'static Con,

    // Returns something that implements ActionExec
)  -> impl ActionExec<()> + '_ {

    // Start building the action sequence macro
    act_nest!(
        // Wrapper, child, child, ...
        ActionSequence::new,
        StartBno055::new(context),
        act_nest! (
            ActionChain::new,
                // Possibly to have a constant yaw change to search???
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(ALIGN_X_SPEED, ALIGN_Y_SPEED, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
        )
    )

    
    // First align to path
    // path_align();

    // Then follow path concurrently
}
