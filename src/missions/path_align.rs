use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionConditional, ActionDataConditional,
            ActionSequence, TupleSecond,
        },
        extra::{CountTrue, NoOp, OutputType, Terminal, ToVec},
        movement::{
            AdjustMovementAngle, LinearYawFromX, OffsetToPose, Stability2Adjust,
            Stability2Movement, Stability2Pos, ZeroMovement,
        },
        vision::{ExtractPosition, MidPoint, VisionNormBottom, VisionNormOffsetBottom},
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
    const DEPTH: f32 = 1.25;
    const PATH_ALIGN_SPEED: f32 = 0.6;

    act_nest!(
        ActionSequence::new,
        ZeroMovement::new(context, DEPTH),
        ActionChain::new(
            VisionNormBottom::<Con, Path, f64>::new(context, Path::default()),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ToVec::new(),
                    ExtractPosition::new(),
                    MidPoint::new(),
                    //NoOp::new(),
                    OffsetToPose::default(),
                    LinearYawFromX::<Stability2Adjust>::default(),
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
                    ),
                    OutputType::<()>::new(),
                ),
                CountTrue::new(3),
            )),
        ),
        Terminal::new(),
    )
}
