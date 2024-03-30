use anyhow::Result;
use async_trait::async_trait;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::act_nest;

use super::{
    action::{
        Action, ActionConcurrent, ActionConditional, ActionExec, ActionMod, ActionSequence,
        RaceAction,
    },
    action_context::{GetControlBoard, GetMainElectronicsBoard},
    basic::DelayAction,
    example::initial_descent,
    meb::WaitArm,
    movement::Descend,
};

/// Looks up at octagon
pub fn look_up_octagon<
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
>(
    context: &Con,
) -> impl ActionExec + '_ {
    ActionSequence::new(initial_descent(context), NoOp::new())
}
