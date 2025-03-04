use std::marker::PhantomData;

use anyhow::bail;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio_serial::SerialStream;

use crate::{
    act_nest, logln,
    missions::{
        action::{ActionChain, ActionConcurrent, ActionSequence, ActionWhile, TupleSecond},
        basic::DelayAction,
        extra::{AlwaysFalse, OutputType},
        movement::{GlobalMovement, GlobalPos, Stability2Movement, Stability2Pos, ZeroMovement},
    },
};

use super::{
    action::{Action, ActionExec},
    action_context::{GetBottomCamMat, GetControlBoard, GetMainElectronicsBoard},
};

pub fn spin<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetBottomCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const GATE_DEPTH: f32 = -1.75;
    const DEPTH: f32 = -1.75;
    const Z_TARGET: f32 = 0.0;
    const FORWARD_SPEED: f32 = 1.0;
    const SPIN_SPEED: f32 = 1.0;

    act_nest!(
        ActionSequence::new,
        ActionChain::new(
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, FORWARD_SPEED, 0.0, 0.0, None, GATE_DEPTH),
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(6.0),
        ActionWhile::new(TupleSecond::new(ActionConcurrent::new(
            act_nest!(
                ActionSequence::new,
                ActionChain::new(
                    GlobalMovement::new(
                        context,
                        GlobalPos::new(0.0, 0.0, Z_TARGET, 0.0, SPIN_SPEED, 0.0),
                    ),
                    OutputType::<()>::new(),
                ),
                ActionChain::new(AlwaysFalse::new(), OutputType::<anyhow::Result<()>>::new(),),
            ),
            SpinCounter::new(4, context)
        ))),
        ZeroMovement::new(context, DEPTH),
        OutputType::<()>::new(),
    )
}

struct SpinCounter<'a, T, U> {
    target: usize,
    half_loops: usize,
    control_board: &'a T,
    _phantom: PhantomData<U>,
}

impl<'a, T, U> SpinCounter<'a, T, U> {
    pub fn new(target: usize, control_board: &'a T) -> Self {
        Self {
            target,
            half_loops: 0,
            control_board,
            _phantom: PhantomData,
        }
    }
}

impl<T, U> Action for SpinCounter<'_, T, U> {}

impl<T: GetControlBoard<U> + Send + Sync, U: AsyncWriteExt + Unpin + Send + Sync>
    ActionExec<anyhow::Result<()>> for SpinCounter<'_, T, U>
{
    async fn execute(&mut self) -> anyhow::Result<()> {
        let cntrl_board = self.control_board.get_control_board();
        if let Some(angles) = cntrl_board.responses().get_angles().await {
            let roll = *angles.roll();
            if self.half_loops % 2 == 0 {
                if roll < -20.0 && roll > -150.0 {
                    logln!("Roll at 0 trigger: {}", roll);
                    self.half_loops += 1;
                    logln!("Loop count: {}", self.half_loops);
                }
            } else if roll < 160.0 && roll > 0.0 {
                logln!("Roll at 1 trigger: {}", roll);
                self.half_loops += 1;
                logln!("Loop count: {}", self.half_loops);
            }
        }

        if self.half_loops < self.target {
            Ok(())
        } else {
            bail!("")
        }
    }
}
