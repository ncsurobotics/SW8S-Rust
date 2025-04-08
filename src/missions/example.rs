use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::act_nest;

use super::{
    action::{Action, ActionConcurrent, ActionConditional, ActionExec, ActionSequence, RaceAction},
    action_context::{GetControlBoard, GetMainElectronicsBoard},
    basic::DelayAction,
    extra::{AlwaysTrue, UnwrapAction},
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

pub fn sequence_conditional<
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    ActionSequence::new(
        ActionSequence::new(WaitArm::new(context), Descend::new(context, -1.0)),
        ActionConditional::new(
            AlwaysTrue::new(),
            WaitArm::new(context),
            UnwrapAction::new(Descend::new(context, -0.5)),
        ),
    )
}

pub fn race_conditional<
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    ActionConditional::new(
        AlwaysTrue::new(),
        WaitArm::new(context),
        RaceAction::new(
            UnwrapAction::new(Descend::new(context, -0.5)),
            DelayAction::new(1.0),
        ),
    )
}

/// Function to demonstrate use of act_nest
pub fn race_many<
    Con: Send + Sync + GetMainElectronicsBoard + GetControlBoard<WriteHalf<SerialStream>>,
>(
    _context: &Con,
) -> impl ActionExec<bool> + '_ {
    ActionSequence::<bool, _, _>::new(
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
