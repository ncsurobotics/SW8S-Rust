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
    meb::WaitArm,
    movement::Descend,
};

/// Example function for Action system
///
/// Runs two nested actions in order: Waiting for arm and descending in
/// parallel, followed by waiting for arm and descending concurrently.
pub fn initial_descent<
    'a,
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
    T: Send + Sync + 'a,
>(
    context: &'a Con,
) -> impl ActionExec<T> + 'a
where
    WaitArm<'a, Con>: ActionExec<T>,
{
    ActionSequence::new(
        ActionConcurrent::new(WaitArm::new(context), Descend::new(context, -0.5)),
        WaitArm::new(context), //ActionConcurrent::new(WaitArm::new(context), Descend::new(context, -1.0)),
    )
}

/// Example function for Action system
///
/// Runs two nested actions in order: Waiting for arm and descending in
/// parallel, followed by waiting for arm and descending concurrently.
pub fn always_wait<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionConditional::new(
        AlwaysTrue::new(),
        WaitArm::new(context),
        Descend::new(context, -0.5),
    )
}

/*
pub fn sequence_conditional<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionSequence::new(
        ActionSequence::new(WaitArm::new(context), Descend::new(context, -1.0)),
        ActionConditional::new(
            AlwaysTrue::new(),
            WaitArm::new(context),
            Descend::new(context, -0.5),
        ),
    )
}
*/

pub fn race_conditional<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionConditional::new(
        AlwaysTrue::new(),
        WaitArm::new(context),
        RaceAction::new(Descend::new(context, -0.5), DelayAction::new(1.0)),
    )
}

/// Function to demonstrate use of act_nest
pub fn race_many<
    'a,
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
    T: Send + Sync + 'a,
>(
    _context: &'a Con,
) -> impl ActionExec<T> + 'a
where
    AlwaysTrue: ActionExec<T>,
{
    ActionSequence::new(
        act_nest!(
            RaceAction::new,
            AlwaysTrue::new(),
            AlwaysTrue::new(),
            AlwaysTrue::new(),
            AlwaysTrue::new(),
            AlwaysTrue::new()
        ),
        AlwaysTrue::new(),
    )
}

#[derive(Debug)]
pub struct AlwaysTrue {}

impl AlwaysTrue {
    pub fn new() -> Self {
        AlwaysTrue {}
    }
}
impl Default for AlwaysTrue {
    fn default() -> Self {
        Self::new()
    }
}

impl Action for AlwaysTrue {}

impl<T: Send + Sync> ActionMod<T> for AlwaysTrue {
    fn modify(&mut self, _input: &T) {}
}

#[async_trait]
impl ActionExec<Result<()>> for AlwaysTrue {
    async fn execute(&mut self) -> Result<()> {
        Ok(())
    }
}
