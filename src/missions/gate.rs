use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{ActionConcurrentSplit, ActionDataConditional},
        basic::descend_depth_and_go_forward,
        extra::{AlwaysFalse, AlwaysTrue, Terminal},
        movement::{
            AdjustType, ClampX, FlipX, InvertX, ReplaceX, SetSideBlue, SetSideRed, SetX, SetY,
        },
        vision::{MidPoint, OffsetClass},
    },
    vision::{
        gate_poles::{GatePoles, Target},
        nn_cv2::{OnnxModel, YoloClass},
        Offset2D,
    },
};

use super::{
    action::{
        wrap_action, ActionChain, ActionConcurrent, ActionExec, ActionMod, ActionSequence,
        ActionWhile, FirstValid, TupleSecond,
    },
    action_context::{GetControlBoard, FrontCamIO, GetMainElectronicsBoard},
    basic::{descend_and_go_forward, DelayAction},
    comms::StartBno055,
    extra::{CountFalse, CountTrue, OutputType},
    movement::{
        AdjustMovementAngle, LinearYawFromX, OffsetToPose, Stability2Adjust, Stability2Movement,
        Stability2Pos, ZeroMovement,
    },
    vision::{DetectTarget, ExtractPosition, VisionNorm, VisionNormOffset},
};

pub fn gate_run_naive<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    let depth: f32 = -1.5;

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
        + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<anyhow::Result<()>> + '_ {
    const TIMEOUT: f32 = 30.0;

    let depth: f32 = -1.40;

    act_nest!(
        ActionSequence::new,
        DelayAction::new(3.0),
        ActionConcurrent::new(
            descend_depth_and_go_forward(context, depth),
            StartBno055::new(context),
        ),
        act_nest!(
            ActionSequence::new,
            adjust_logic(context, depth, CountTrue::new(4)),
            adjust_logic(context, depth, CountFalse::new(4)),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            DelayAction::new(3.0),
            ZeroMovement::new(context, depth),
        ),
    )
}

pub fn gate_run_coinflip<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<anyhow::Result<()>> + '_ {
    const TIMEOUT: f32 = 30.0;

    let depth: f32 = -1.6;

    act_nest!(
        ActionSequence::new,
        ActionConcurrent::new(
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            StartBno055::new(context),
        ),
        act_nest!(
            ActionSequence::new,
            adjust_logic(context, depth, CountTrue::new(4)),
            // adjust_logic(context, depth, CountFalse::new(10)),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            ActionWhile::new(act_nest!(
                ActionChain::new,
                VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default()),
                act_nest!(
                    wrap_action(ActionConcurrent::new, FirstValid::new),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Middle),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
                ),
                CountFalse::new(5),
            )),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            DelayAction::new(3.0),
            ZeroMovement::new(context, depth),
        ),
    )
}

pub fn adjust_logic<
    'a,
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + FrontCamIO,
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
            TupleSecond::new(ActionConcurrent::new(
                ActionDataConditional::new(
                    //act_nest!(
                    //wrap_action(ActionConcurrent::new, FirstValid::new),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                    //DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(
                    //Target::Middle
                    //),
                    //),
                    ActionSequence::new(SetSideBlue::new(), Terminal::new()),
                    ActionDataConditional::new(
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                        ActionSequence::new(SetSideRed::new(), Terminal::new()),
                        Terminal::new(),
                    ),
                ),
                ActionDataConditional::new(
                    act_nest!(
                        wrap_action(ActionConcurrent::new, FirstValid::new),
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(
                            Target::Middle
                        ),
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                    ),
                    act_nest!(
                        ActionConcurrent::new,
                        act_nest!(
                            ActionChain::new,
                            OffsetClass::new(Target::Middle, Offset2D::<f64>::new(-0.05, 0.0)),
                            //OffsetClass::new(Target::Blue, Offset2D::<f64>::new(-0.1, 0.0)),
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::default(),
                            LinearYawFromX::<Stability2Adjust>::new(5.0),
                            ClampX::new(0.2),
                            SetY::<Stability2Adjust>::new(AdjustType::Adjust(0.02)),
                            FlipX::default(),
                        ),
                        AlwaysTrue::new(),
                    ),
                    ActionDataConditional::new(
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
                        act_nest!(
                            ActionConcurrent::new,
                            act_nest!(
                                ActionChain::new,
                                ExtractPosition::new(),
                                MidPoint::new(),
                                OffsetToPose::default(),
                                InvertX::new(),
                                LinearYawFromX::<Stability2Adjust>::new(-7.0),
                                //ClampX::new(0.8),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(0.2)),
                                ReplaceX::new(),
                            ),
                            AlwaysTrue::new(),
                        ),
                        ActionConcurrent::new(
                            act_nest!(
                                ActionSequence::new,
                                Terminal::new(),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(0.4)),
                                SetX::<Stability2Adjust>::new(AdjustType::Replace(0.0)),
                            ),
                            AlwaysFalse::new(),
                        ),
                    ),
                ),
            )),
            TupleSecond::new(ActionConcurrentSplit::new(
                act_nest!(
                    ActionChain::new,
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
        + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    let depth: f32 = -1.0;
    adjust_logic(context, depth, CountTrue::new(3))
}
