use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::movement::StripY,
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
            adjust_logic(context, depth, CountTrue::new(3)),
            adjust_logic(context, depth, CountFalse::new(10)),
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
    X: 'a
        + ActionMod<Option<Vec<VisualDetection<YoloClass<Target>, Offset2D<f64>>>>>
        + ActionExec<anyhow::Result<()>>,
>(
    context: &'a Con,
    depth: f32,
    end_condition: X,
) -> impl ActionExec<()> + 'a {
    const GATE_TRAVERSAL_SPEED: f32 = 0.5;

    println!("Fire adjust logic");
    ActionWhile::new(ActionChain::new(
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default()),
        ActionChain::new(
            act_nest!(
                wrap_action(ActionConcurrent::new, FirstValid::new),
                DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Earth),
                DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Abydos),
                DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::LargeGate),
                DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
            ),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ToVec::new(),
                    ExtractPosition::new(),
                    Average::new(),
                    OffsetToPose::default(),
                    LinearYawFromX::<Stability2Adjust>::default(),
                    StripY::default(),
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, GATE_TRAVERSAL_SPEED, 0.0, 0.0, None, depth)
                    ),
                    OutputType::<()>::new()
                ),
                end_condition,
            )),
        ),
    ))
}
