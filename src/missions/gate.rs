use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{ActionConcurrentSplit, ActionDataConditional},
        extra::{AlwaysFalse, AlwaysTrue, NoOp, Terminal},
        movement::{AdjustType, ConfidenceY, DefaultGen, FlipX, FlipYaw, SetY, StripY},
        vision::{MidPoint, OffsetClass},
    },
    vision::{
        gate_poles::{GatePoles, Target},
        nn_cv2::{OnnxModel, YoloClass},
        Offset2D, VisualDetection,
    },
};

use super::{
    action::{
        wrap_action, ActionChain, ActionConcurrent, ActionExec, ActionMod, ActionSequence,
        ActionWhile, FirstValid, TupleSecond,
    },
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::descend_and_go_forward,
    comms::StartBno055,
    extra::{CountFalse, CountTrue, OutputType, ToVec},
    movement::{
        AdjustMovementAngle, LinearYawFromX, OffsetToPose, Stability2Adjust, Stability2Movement,
        Stability2Pos, ZeroMovement,
    },
    vision::{Average, DetectTarget, ExtractPosition, VisionNorm, VisionNormOffset},
};

pub fn gate_run_naive<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    let depth: f32 = -1.0;

    ActionSequence::new(
        ActionConcurrent::new(descend_and_go_forward(context), StartBno055::new(context)),
        ActionSequence::new(
            ActionWhile::new(ActionChain::new(
                VisionNormOffset::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::default(),
                ),
                TupleSecond::new(ActionConcurrent::new(
                    AdjustMovementAngle::new(context, depth),
                    CountTrue::new(3),
                )),
            )),
            ActionWhile::new(ActionChain::new(
                VisionNormOffset::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::default(),
                ),
                TupleSecond::new(ActionConcurrent::new(
                    AdjustMovementAngle::new(context, depth),
                    CountFalse::new(10),
                )),
            )),
        ),
    )
}

pub fn gate_run_complex<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec<anyhow::Result<()>> + '_ {
    let depth: f32 = -1.0;

    ActionSequence::new(
        ActionConcurrent::new(descend_and_go_forward(context), StartBno055::new(context)),
        act_nest!(
            ActionSequence::new,
            adjust_logic(context, depth, CountTrue::new(4)),
            adjust_logic(context, depth, CountFalse::new(4)),
            ZeroMovement::new(context, depth)
        ),
    )
}

pub fn adjust_logic<
    'a,
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
    X: 'a + ActionMod<bool> + ActionExec<anyhow::Result<()>>,
>(
    context: &'a Con,
    depth: f32,
    end_condition: X,
) -> impl ActionExec<()> + 'a {
    const GATE_TRAVERSAL_SPEED: f32 = 0.2;

    ActionWhile::new(ActionChain::new(
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default()),
        ActionChain::new(
            ActionDataConditional::new(
                act_nest!(
                    wrap_action(ActionConcurrent::new, FirstValid::new),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Gate),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Middle),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                ),
                ActionConcurrent::new(
                    act_nest!(
                        ActionChain::new,
                        OffsetClass::new(Target::Middle, Offset2D::<f64>::new(-0.1, 0.0)),
                        ExtractPosition::new(),
                        MidPoint::new(),
                        OffsetToPose::default(),
                        LinearYawFromX::<Stability2Adjust>::default(),
                        SetY::<Stability2Adjust>::new(AdjustType::Adjust(0.05)),
                        FlipX::default(),
                    ),
                    AlwaysTrue::new(),
                ),
                ActionConcurrent::new(
                    ActionDataConditional::new(
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
                        act_nest!(
                            ActionChain::new,
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::default(),
                            LinearYawFromX::<Stability2Adjust>::default(),
                            FlipYaw::default(),
                            SetY::<Stability2Adjust>::new(AdjustType::Adjust(-0.1)),
                        ),
                        ActionSequence::new(
                            Terminal::new(),
                            SetY::<Stability2Adjust>::new(AdjustType::Replace(0.2)),
                        ),
                    ),
                    AlwaysFalse::new(),
                ),
            ),
            TupleSecond::new(ActionConcurrentSplit::new(
                ActionChain::new(
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, GATE_TRAVERSAL_SPEED, 0.0, 0.0, None, depth),
                    ),
                    OutputType::<()>::new(),
                ),
                end_condition,
            )),
        ),
    ))
}

pub fn gate_run_testing<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    let depth: f32 = -1.0;
    adjust_logic(context, depth, CountTrue::new(3))
}
