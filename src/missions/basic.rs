use crate::logln;

use super::{
    action::{Action, ActionChain, ActionExec, ActionSequence},
    action_context::{GetControlBoard, GetMainElectronicsBoard},
    extra::OutputType,
    meb::WaitArm,
    movement::{Descend, Stability2Movement, Stability2Pos, StraightMovement, ZeroMovement},
};

use tokio::{
    io::WriteHalf,
    time::{sleep, Duration},
};
use tokio_serial::SerialStream;

#[derive(Debug, Clone)]
pub struct DelayAction {
    delay: f32, // delay in seconds before the next action occurs.
}

impl Action for DelayAction {}

impl ActionExec<()> for DelayAction {
    async fn execute(&mut self) {
        logln!("BEGIN sleep for {} seconds", self.delay);
        sleep(Duration::from_secs_f32(self.delay)).await;
        logln!("END sleep for {} seconds", self.delay);
    }
}

impl DelayAction {
    pub const fn new(delay: f32) -> Self {
        Self { delay }
    }
}

/**
 *
 * descends and goes forward for a certain duration
 *
 **/
pub fn descend_and_go_forward<
    'a,
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard,
    T: Send + Sync,
>(
    context: &'a Con,
) -> impl ActionExec<T> + 'a
where
    ZeroMovement<'a, Con>: ActionExec<T>,
{
    let depth: f32 = -1.5;

    // time in seconds that each action will wait until before continuing onto the next action.
    let dive_duration = 2.0;
    let forward_duration = 0.0;
    ActionSequence::new(
        WaitArm::new(context),
        ActionSequence::new(
            ActionSequence::new(
                Descend::new(context, depth),
                DelayAction::new(dive_duration),
            ),
            ActionSequence::new(
                ActionSequence::new(
                    StraightMovement::new(context, depth, true),
                    DelayAction::new(forward_duration),
                ),
                ZeroMovement::new(context, depth),
            ),
        ),
    )
}

pub fn descend_depth_and_go_forward<
    'a,
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard,
    T: Send + Sync,
>(
    context: &'a Con,
    depth: f32,
) -> impl ActionExec<T> + 'a
where
    ZeroMovement<'a, Con>: ActionExec<T>,
{
    // time in seconds that each action will wait until before continuing onto the next action.
    let dive_duration = 4.0;
    let forward_duration = 2.0;
    ActionSequence::new(
        WaitArm::new(context),
        ActionSequence::new(
            ActionSequence::new(
                ActionChain::new(
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, depth),
                    ),
                    OutputType::<()>::new(),
                ),
                DelayAction::new(dive_duration),
            ),
            ActionSequence::new(
                ActionSequence::new(
                    StraightMovement::new(context, depth, true),
                    DelayAction::new(forward_duration),
                ),
                ZeroMovement::new(context, depth),
            ),
        ),
    )
}
