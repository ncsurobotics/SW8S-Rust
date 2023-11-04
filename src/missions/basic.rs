use crate::{
    video_source::MatSource,
    vision::{buoy::Buoy, nn_cv2::OnnxModel},
};

use super::{
    action::{
        Action, ActionChain, ActionConcurrent, ActionExec, ActionSequence, ActionWhile, TupleSecond,
    },
    action_context::{GetControlBoard, GetMainElectronicsBoard},
    comms::StartBno055,
    example::AlwaysTrue,
    meb::WaitArm,
    movement::StraightMovement,
    movement::ZeroMovement,
    movement::{AdjustMovement, Descend},
    vision::VisionNormOffset,
};
use async_trait::async_trait;
use tokio::{
    io::WriteHalf,
    time::{sleep, Duration},
};
use tokio_serial::SerialStream;

#[derive(Debug)]
pub struct DelayAction {
    delay: f32, // delay in seconds before the next action occurs.
}

impl Action for DelayAction {}

#[async_trait]
impl ActionExec for DelayAction {
    type Output = ();
    async fn execute(&mut self) -> Self::Output {
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
    T: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard,
>(
    context: &T,
) -> impl ActionExec + '_ {
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

pub fn gate_run<
    T: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + MatSource,
>(
    context: &T,
) -> impl ActionExec + '_ {
    let depth: f32 = -1.0;
    println!("INNER MODEL LOAD");
    let model = Buoy::default();
    println!("OUTER MODEL LOAD");

    ActionSequence::new(
        ActionConcurrent::new(descend_and_go_forward(context), StartBno055::new(context)),
        ActionWhile::new(TupleSecond::new(ActionSequence::new(
            ActionChain::new(
                VisionNormOffset::<T, Buoy<OnnxModel>, f64>::new(context, model),
                AdjustMovement::new(context, depth),
            ),
            AlwaysTrue::new(),
        ))),
    )
}
