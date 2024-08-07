use std::marker::PhantomData;

use anyhow::bail;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionConditional, ActionDataConditional,
            ActionSequence, ActionWhile, TupleSecond,
        },
        basic::DelayAction,
        extra::{AlwaysFalse, CountTrue, NoOp, OutputType, Terminal, ToVec},
        movement::{
            AdjustMovementAngle, GlobalMovement, GlobalPos, LinearYawFromX, OffsetToPose,
            Stability2Adjust, Stability2Movement, Stability2Pos, ZeroMovement,
        },
        vision::{ExtractPosition, MidPoint, VisionNormBottom, VisionNormOffsetBottom},
    },
    vision::path::Path,
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
    const DEPTH: f32 = 0.5;
    const Z_TARGET: f32 = 0.0;
    const FORWARD_SPEED: f32 = 0.0;
    const SPIN_SPEED: f32 = 1.0;

    act_nest!(
        ActionSequence::new,
        ZeroMovement::new(context, DEPTH),
        DelayAction::new(1.0),
        ActionWhile::new(TupleSecond::new(ActionConcurrent::new(
            act_nest!(
                ActionSequence::new,
                ActionChain::new(
                    GlobalMovement::new(
                        context,
                        GlobalPos::new(0.0, FORWARD_SPEED, Z_TARGET, SPIN_SPEED, 0.0, 0.0),
                    ),
                    OutputType::<()>::new(),
                ),
                ActionChain::new(AlwaysFalse::new(), OutputType::<anyhow::Result<()>>::new(),),
            ),
            SpinCounter::new(4, context)
        ))),
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
            if self.half_loops % 2 == 0 {
                if angles.roll().abs() > 180.0 {
                    self.half_loops += 1;
                }
            } else if angles.roll().abs() < 180.0 {
                self.half_loops += 1;
            }
        }

        if self.half_loops < self.target {
            Ok(())
        } else {
            bail!("")
        }
    }
}
