use tokio::io::WriteHalf;
use tokio_serial::SerialStream;
use crate::missions;
use missions::action_context::GetBottomCamMat;
use crate::missions::path_align::path_align;
use std::default::Default;
use crate::missions::gate::adjust_logic;
use crate::missions::action::Action;
use crate::comms::meb::MainElectronicsBoard; // Import MainElectronicsBoard
use crate::comms::meb::MebCmd; // Import MebCmd
use super::{action::ActionExec, action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard}};
use crate::vision::bins::Bin;
use crate::vision::nn_cv2::YoloClass;
use crate::vision::Offset2D;
use crate::vision::DrawRect2d;

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
    vision::bins::Target,
    POOL_YAW_SIGN,
};

#[derive(Debug)]
pub struct DropObject<'a, T> {
    meb: &'a T,
}

impl<'a, T> DropObject<'a, T> {
    pub fn new(meb: &'a T) -> Self {
        Self { meb }
    }
}

impl<T> Action for DropObject<'_, T> {}

impl<T: GetMainElectronicsBoard> ActionExec<()> for DropObject<'_, T> {
    async fn execute<'a>(&'a mut self) {
        let send_cmd = |meb: &'a MainElectronicsBoard<WriteHalf<SerialStream>>, cmd| async move {
            if let Err(e) = meb.send_msg(cmd).await {
            logln!("{:#?} failure: {:#?}", cmd, e);
            } else {
            logln!("{:#?} success", cmd);
            }
        };

        let meb = self.meb.get_main_electronics_board();
        for _ in 0..3 {
            send_cmd(meb, MebCmd::D1Trig).await;
        }
    }
}

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
        + GetBottomCamMat
        + std::fmt::Display
        + std::cmp::PartialEq,
>(
    context: &'static Con,
) -> impl ActionExec<anyhow::Result<()>> + '_ {
    // Part 1: Move to the location to drop
    ActionSequence::new(
        ActionConcurrent::new(
            // Step 1: Align with the path
            path_align(context),
            // Step 2: Initialize the BNO055 sensor
            StartBno055::new(context),
        ),
        // Step 3: Try to move with the path
        act_nest!(
            ActionSequence::new,
            adjust_logic(context, DEPTH, CountTrue::new(4)),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(ALIGN_X_SPEED, ALIGN_Y_SPEED, 0.0, 0.0, None, DEPTH),
                ),
                OutputType::<()>::default()
            ),
            // Step 4: Vision part to detect the target and drop the item
            ActionSequence::new(
                Vision::new(context, Bin::new("bins_640.onnx", 224, 0.5).unwrap()), // Detect objects using the bottom camera
                ActionSequence::new(
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::SawFish), // Provide the additional generic arguments
                    ActionSequence::new(
                        SizeUnder::<Target, DrawRect2d>::new(0.5), // Ensure the target is within the size threshold
                        DropObject::new(context), // Drop the item
                    ),
                ),
            ),
            // Step 5: Try to stop at this target after finding it
            ActionSequence::new(
                DelayAction::new(3.0),
                ZeroMovement::new(context, DEPTH),
            ),
        )
    )
}