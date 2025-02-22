use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            TupleSecond,
        },
        extra::{CountTrue, OutputType, Terminal, ToVec},
        movement::{
            LinearYawFromX, OffsetToPose, Stability2Adjust, Stability2Movement, Stability2Pos,
            ZeroMovement,
        },
        vision::{DetectTarget, ExtractPosition, MidPoint, VisionNormBottom},
    },
    vision::path::Path,
};

use super::{
    action::ActionExec,
    action_context::{GetBottomCamMat, GetControlBoard, GetMainElectronicsBoard},
};

pub fn path_align<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetBottomCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const DEPTH: f32 = -1.25;
    const PATH_ALIGN_SPEED: f32 = 0.3;

    act_nest!(
        ActionSequence::new,
        ZeroMovement::new(context, DEPTH),
        DelayAction::new(2),
        ActionWhile::new(ActionChain::new(
            VisionNormBottom::<Con, Path, f64>::new(context, Path::default()),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ActionDataConditional::new(
                        DetectTarget::new(true),
                        act_nest!(
                            ActionChain::new,
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::default(),
                            LinearYawFromX::<Stability2Adjust>::default(),
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 30.0, 0.0, None, DEPTH),
                            ),
                        ),
                        act_nest!(
                            ActionSequence::new,
                            Terminal::new(),
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 30.0, 0.0, None, DEPTH),
                            ),
                        ),
                    ),
                    OutputType::<()>::new(),
                ),
                ActionChain::new(DetectTarget::new(true), CountTrue::new(5)),
            )),
        )),
        Terminal::new(),
    )
}
