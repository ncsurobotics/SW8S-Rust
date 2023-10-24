use super::{
    action::{Action, ActionExec, ActionConcurrent, ActionParallel, ActionSequence},
    meb::WaitArm,
    movement::Descend,
    movement::StraightMovement,
    movement::ZeroMovement
};
use async_trait::async_trait;
use tokio::time::{sleep, Duration};


#[derive(Debug)]
struct DelayAction {
    delay: f32 // delay in seconds before the next action occurs.
}

impl Action for DelayAction {}

#[async_trait]
impl ActionExec<()> for DelayAction {
    async fn execute(self) -> (){
        sleep(Duration::from_secs_f32(self.delay)).await;
    }
}

impl DelayAction {
    pub const fn new(delay: f32) -> Self {
        Self {
            delay
        }
    }
}




/// Example function for Action system
///
/// Runs two nested actions in order: Waiting for arm and descending in
/// parallel, followed by waiting for arm and descending concurrently.
fn initial_descent<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionSequence::<T, T, _, _>::new(
        ActionParallel::<T, T, _, _>::new(
            WaitArm::new(context), 
        Descend::new(context, -0.5)
    ),
        ActionConcurrent::<T, T, _, _>::new(
            WaitArm::new(context), 
        Descend::new(context, -1.0)
    ),
    )
}

/**
 * 
 * descends and goes forward for a certain duration 
 * 
 **/

fn descend_and_go_forward<T: Send + Sync>(context: &T) -> impl Action + '_ {
    let depth: f32 = -1.0; 

    // time in seconds that each action will wait until before continuing onto the next action. 
    let dive_duration = 5.0; 
    let forward_duration = 10.0;
    ActionSequence::<T, T, _, _>::new(
        WaitArm::new(context),
        ActionSequence::<T,T,_,_>::new(
            ActionSequence::<T,T,_,_>::new(
                Descend::new(context, depth),
                DelayAction::new(dive_duration)
            ), 
            ActionSequence::<T,T,_,_>::new(
                ActionSequence::<T,T,_,_>::new(
                    StraightMovement::new(context, depth, true),
                    DelayAction::new(forward_duration)
                ),
                ZeroMovement::new(context, depth)
            )
        )
    )
}