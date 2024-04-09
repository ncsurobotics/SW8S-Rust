use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{act_nest, vision::path::Path};

use super::{
    action::{ActionExec, ActionSequence},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    example::initial_descent,
    extra::NoOp,
    movement::{Stability2Movement, Stability2Pos},
    vision::VisionNorm,
};

/// Looks up at octagon
pub fn look_up_octagon<
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const DEPTH: f32 = -2.5;

    ActionSequence::<(), _, _>::new(
        act_nest!(
            ActionSequence::<(), _, _>::new,
            initial_descent(context),
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, 0.0, 90.0, 0.0, None, DEPTH),
            ),
            DelayAction::new(5.0),
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, 0.0, 180.0, 0.0, None, DEPTH),
            ),
            DelayAction::new(5.0),
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, 0.0, 180.0, 0.0, None, DEPTH),
            ),
        ),
        DelayAction::new(30.0),
    )
}

pub fn stub<
    Con: Send
        + Sync
        + GetMainElectronicsBoard
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    ActionSequence::new(
        VisionNorm::<Con, Path, f64>::new(context, Path::default()),
        NoOp::new(),
    )
}
