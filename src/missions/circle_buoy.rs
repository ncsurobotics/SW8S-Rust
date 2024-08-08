use crate::{
    act_nest,
    missions::{
        action::{ActionChain, ActionConcurrent, ActionWhile, TupleSecond},
        basic::descend_and_go_forward,
        extra::{AlwaysTrue, CountTrue, OutputType, ToVec, Transform},
        movement::{
            aggressive_yaw_from_x, AdjustType, CautiousConstantX, ConstYaw, Descend, FlatX,
            LinearYawFromX, MinYaw, OffsetToPose, SetX, SideMult, Stability1Adjust,
            Stability1Movement, Stability1Pos, Stability2Adjust, Stability2Movement, Stability2Pos,
            StripY,
        },
        vision::{Average, DetectTarget, ExtractPosition, VisionNorm},
    },
    vision::{
        buoy_model::{BuoyModel, Target},
        nn_cv2::{OnnxModel, YoloClass},
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
    const BUOY_X_SPEED: f32 = -0.0;
    const BUOY_Y_SPEED: f32 = 0.0;
    const DEPTH: f32 = -1.0;
    //const NUM_MODEL_THREADS: NonZeroUsize = nonzero!(4_usize);

    act_nest!(
        ActionSequence::new,
        descend_and_go_forward(context),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            VisionNorm::<Con, BuoyModel<OnnxModel>, f64>::new(context, BuoyModel::default()),
            DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Buoy),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ToVec::new(),
                    ExtractPosition::new(),
                    Average::new(),
                    //BoxToPose::default(),
                    OffsetToPose::default(),
                    LinearYawFromX::<Stability1Adjust>::new(4.0),
                    CautiousConstantX::<Stability1Adjust>::new(-0.3),
                    StripY::<Stability1Adjust>::new(),
                    //FlipYaw::<Stability1Adjust>::new(),
                    //MinYaw::<Stability1Adjust>::new(-3.0),
                    MinYaw::<Stability1Adjust>::new(12.0),
                    Stability1Movement::new(
                        context,
                        Stability1Pos::new(BUOY_X_SPEED, BUOY_Y_SPEED, 0.0, 0.0, 0.0, DEPTH)
                    ),
                    OutputType::<()>::new()
                ),
                AlwaysTrue::default(),
            )),
        )),
        OutputType::<()>::new()
    )
}

pub fn buoy_circle_sequence_blind<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    const BUOY_X_SPEED: f32 = -0.4;
    const BUOY_Y_SPEED: f32 = 0.15;
    const BUOY_YAW_SPEED: f32 = -14.0;
    const DEPTH: f32 = -1.5;
    const DESCEND_WAIT_DURATION: f32 = 3.0;
    const CIRCLE_COUNT: u32 = 28;

    act_nest!(
        ActionSequence::new,
        Descend::new(context, DEPTH),
        DelayAction::new(DESCEND_WAIT_DURATION),
        ActionWhile::new(act_nest!(
            ActionSequence::new,
            act_nest!(
                ActionChain::new,
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(BUOY_YAW_SPEED)),
                SetX::<Stability2Adjust>::new(AdjustType::Replace(BUOY_X_SPEED)),
                SideMult::new(),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, BUOY_Y_SPEED, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new()
            ),
            DelayAction::new(1.0),
            ActionChain::<bool, _, _>::new(AlwaysTrue::default(), CountTrue::new(CIRCLE_COUNT)),
        )),
        ZeroMovement::new(context, DEPTH),
        OutputType::<()>::new()
    )
}
