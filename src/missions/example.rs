use async_trait::async_trait;

use super::{
    action::{
        Action, ActionConcurrent, ActionConditional, ActionExec, ActionParallel, ActionSequence,
    },
    meb::WaitArm,
    movement::Descend,
};

/// Example function for Action system
///
/// Runs two nested actions in order: Waiting for arm and descending in
/// parallel, followed by waiting for arm and descending concurrently.
pub fn initial_descent<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionSequence::<T, T, _, _>::new(
        ActionParallel::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -0.5)),
        ActionConcurrent::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -1.0)),
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

#[derive(Debug)]
struct AlwaysTrue {}

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
