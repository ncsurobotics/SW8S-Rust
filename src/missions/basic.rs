use super::{
    action::{Action, ActionExec, ActionSequence},
    action_context::{GetControlBoard, GetMainElectronicsBoard},
    meb::WaitArm,
    movement::{Descend, StraightMovement, ZeroMovement},
};

use async_trait::async_trait;
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

#[async_trait]
impl ActionExec<()> for DelayAction {
    async fn execute(&mut self) -> () {
        println!("BEGIN sleep for {} seconds", self.delay);
        sleep(Duration::from_secs_f32(self.delay)).await;
        println!("END sleep for {} seconds", self.delay);
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
    let depth: f32 = -1.0;

    // time in seconds that each action will wait until before continuing onto the next action.
    let dive_duration = 5.0;
    let forward_duration = 5.0;
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
