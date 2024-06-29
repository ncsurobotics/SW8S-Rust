use crate::{
    act_nest,
    missions::{
        action::{ActionChain, ActionConcurrent, ActionWhile, TupleSecond},
        basic::descend_and_go_forward,
        extra::{AlwaysTrue, OutputType, ToVec, Transform},
        movement::{
            aggressive_yaw_from_x, FlatX, FlipYaw, LinearYawFromX, OffsetToPose, Stability2Adjust,
            Stability2Movement, Stability2Pos, StripX, StripY,
        },
        vision::{
            Average, DetectTarget, ExtractPosition, OffsetClass, ToOffset, VisionNorm,
            VisionPipelinedNorm,
        },
    },
    vision::{
        buoy_model::{BuoyModel, Target},
        nn_cv2::{ModelPipelined, OnnxModel, YoloClass},
        path::{Path, Yuv},
        Offset2D,
    },
};

use super::{
    action::{ActionExec, ActionSequence},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    movement::ZeroMovement,
};

use nonzero::nonzero;
use opencv::core::Size;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

pub fn buoy_circle_sequence<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const DEPTH: f32 = -0.5;

    let delay_s = 1.0;
    // Create a DelayAction with hardcoded delay
    let delay_action = DelayAction::new(delay_s);

    // Create the inner ActionSequence
    ActionSequence::new(
        ZeroMovement::new(context, DEPTH),
        ActionSequence::new(
            delay_action.clone(),
            ActionWhile::new(ActionSequence::new(
                act_nest!(
                    ActionChain::new,
                    VisionNorm::<Con, Path, f64>::new(
                        context,
                        Path::new(
                            (Yuv { y: 0, u: 0, v: 128 })..=(Yuv {
                                y: 255,
                                u: 127,
                                v: 255,
                            }),
                            20.0..=800.0,
                            10,
                            Size::from((400, 300)),
                            3,
                        )
                    ),
                    DetectTarget::<bool, bool, Offset2D<f64>>::new(true),
                    ToVec::new(),
                    ExtractPosition::new(),
                    Average::new(),
                    OffsetToPose::default(),
                    Transform::new(Stability2Adjust::default(), |input| aggressive_yaw_from_x(
                        input, 40.0
                    )),
                    StripY::default(),
                    FlatX::default(),
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)
                    ),
                    OutputType::<()>::new()
                ),
                AlwaysTrue::new(),
            )),
        ),
    )
}

pub fn buoy_circle_sequence_model<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    const BUOY_X_SPEED: f32 = -0.2;
    const BUOY_Y_SPEED: f32 = 0.2;
    const DEPTH: f32 = -1.0;

    act_nest!(
        ActionSequence::new,
        descend_and_go_forward(context),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            VisionPipelinedNorm::new(context, BuoyModel::default()),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ToVec::new(),
                    ToOffset::new(),
                    Average::new(),
                    OffsetToPose::default(),
                    LinearYawFromX::<Stability2Adjust>::new(60.0),
                    FlipYaw::default(),
                    StripY::default(),
                    StripX::default(),
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(BUOY_X_SPEED, BUOY_Y_SPEED, 0.0, 0.0, None, DEPTH)
                    ),
                    OutputType::<()>::new()
                ),
                AlwaysTrue::default(),
            )),
        )),
        OutputType::<()>::new()
    )
}
