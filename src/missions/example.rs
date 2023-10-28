use anyhow::Result;
use async_trait::async_trait;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use super::{
    action::{
        Action, ActionConcurrent, ActionConditional, ActionExec, ActionParallel, ActionSequence,
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
    T: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
>(
    context: &T,
) -> impl ActionExec<(((), Result<()>), ())> + '_ {
    ActionSequence::<T, T, _, _>::new(
        ActionConcurrent::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -0.5)),
        WaitArm::new(context), //ActionConcurrent::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -1.0)),
    )
}

/// Example function for Action system
///
/// Runs two nested actions in order: Waiting for arm and descending in
/// parallel, followed by waiting for arm and descending concurrently.
pub fn always_wait<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionConditional::<T, _, _, _>::new(
        AlwaysTrue::new(),
        WaitArm::new(context),
        Descend::new(context, -0.5),
    )
}

pub fn sequence_conditional<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionSequence::<T, T, _, _>::new(
        ActionSequence::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -1.0)),
        ActionConditional::<T, _, _, _>::new(
            AlwaysTrue::new(),
            WaitArm::new(context),
            Descend::new(context, -0.5),
        ),
    )
}

pub fn race_conditional<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionConditional::<T, _, _, _>::new(
        AlwaysTrue::new(),
        WaitArm::new(context),
        RaceAction::new(Descend::new(context, -0.5), DelayAction::new(1.0)),
    )
}

#[derive(Debug)]
pub struct AlwaysTrue {}

impl AlwaysTrue {
    pub fn new() -> Self {
        AlwaysTrue {}
    }
}

impl Action for AlwaysTrue {}

#[async_trait]
impl ActionExec<bool> for AlwaysTrue {
    async fn execute(&mut self) -> bool {
        true
    }
}
