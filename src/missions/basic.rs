use crate::{
    video_source::MatSource,
    vision::{gate::Gate, nn_cv2::OnnxModel, RelPos},
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
use anyhow::Result;
use async_trait::async_trait;
use tokio::{
    io::{AsyncWriteExt, WriteHalf},
    time::{sleep, Duration},
};
use tokio_serial::SerialStream;

#[derive(Debug)]
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

pub fn descend_and_go_forward<T: Send + Sync>(context: &T) -> impl Action + '_ {
    let depth: f32 = -1.0;

    // time in seconds that each action will wait until before continuing onto the next action.
    let dive_duration = 5.0;
    let forward_duration = 1.0;
    ActionSequence::<T, T, _, _>::new(
        WaitArm::new(context),
        ActionSequence::<T, T, _, _>::new(
            ActionSequence::<T, T, _, _>::new(
                Descend::new(context, depth),
                DelayAction::new(dive_duration),
            ),
            ActionSequence::<T, T, _, _>::new(
                ActionSequence::<T, T, _, _>::new(
                    StraightMovement::new(context, depth, true),
                    DelayAction::new(forward_duration),
                ),
                ZeroMovement::new(context, depth),
            ),
        ),
    )
}

pub fn descend_and_go_forward_temp<
    T: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard,
>(
    context: &T,
) -> impl ActionExec<((), ((Result<()>, ()), ((Result<()>, ()), Result<()>)))> + '_ {
    let depth: f32 = -1.0;

    // time in seconds that each action will wait until before continuing onto the next action.
    let dive_duration = 5.0;
    let forward_duration = 5.0;
    ActionSequence::<T, T, _, _>::new(
        WaitArm::new(context),
        ActionSequence::<T, T, _, _>::new(
            ActionSequence::<T, T, _, _>::new(
                Descend::new(context, depth),
                DelayAction::new(dive_duration),
            ),
            ActionSequence::<T, T, _, _>::new(
                ActionSequence::<T, T, _, _>::new(
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
) -> impl ActionExec<(
    (
        ((), ((Result<()>, ()), ((Result<()>, ()), Result<()>))),
        Result<()>,
    ),
    (),
)> + '_ {
    let depth: f32 = -1.0;

    ActionSequence::<T, T, _, _>::new(
        ActionConcurrent::<T, T, _, _>::new(
            descend_and_go_forward_temp(context),
            ZeroMovement::new(context, depth),
        ),
        ActionWhile::new(TupleSecond::new(ActionSequence::<T, T, _, _>::new(
            ActionChain::<_, _, _>::new(
                VisionNormOffset::<T, Gate<OnnxModel>, f64>::new(context, Gate::default()),
                AdjustMovement::new(context, depth),
            ),
            AlwaysTrue::new(),
        ))),
    )
}
