use crate::{
    act_nest,
    missions::{
        action::ActionChain,
        extra::ToVec,
        vision::{Average, ExtractPosition, VisionNorm},
    },
    vision::{
        buoy,
        nn_cv2::OnnxModel,
        path::{BuoyPCA, Path},
        Offset2D, VisualDetection, VisualDetector,
    },
};

use super::{
    action::{Action, ActionExec, ActionMod, ActionSequence, ActionWhile},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    movement::{StraightMovement, ZeroMovement},
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

pub struct CircleBuoy<'a, T> {
    context: &'a T,
    target_depth: f32,
    lateral_power: f32,
    rotation: f32,
}

impl<'a, T> CircleBuoy<'a, T> {
    pub fn new(context: &'a T, target_depth: f32, forward_power: f32, lateral_power: f32) -> Self {
        CircleBuoy {
            context,
            target_depth,
            lateral_power,
            rotation: 0.0,
        }
    }
}

impl<T> Action for CircleBuoy<'_, T> {}

#[async_trait]
impl<T> ActionExec<()> for CircleBuoy<'_, T>
where
    T: GetControlBoard<WriteHalf<SerialStream>> + GetFrontCamMat + Sync + Unpin,
{
    async fn execute(&mut self) {
        let _ = self
            .context
            .get_control_board()
            .stability_2_speed_set(
                0.0,
                self.lateral_power,
                0.0,
                0.0,
                self.rotation,
                self.target_depth,
            )
            .await;
    }
}

impl<T: Send + Sync> ActionMod<Option<Offset2D<f64>>> for CircleBuoy<'_, T> {
    fn modify(&mut self, input: &Option<Offset2D<f64>>) {
        if let Some(offset) = input.map(|input| *input.y() as f32) {
            self.rotation = 30.0 * offset;
        }
    }
}

pub fn buoy_circle_sequence<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const DEPTH: f32 = -1.0;

    let lateral_power = 0.3;
    let delay_s = 5.0;
    // Create a DelayAction with hardcoded delay
    let delay_action = DelayAction::new(delay_s);

    // Create the inner ActionSequence
    act_nest!(
        ActionSequence::new,
        delay_action.clone(),
        ZeroMovement::new(context, DEPTH),
        ActionSequence::new(delay_action, ZeroMovement::new(context, DEPTH)),
        act_nest!(
            ActionChain::new,
            VisionNorm::<Con, Path, f64>::new(context, Path::default()),
            ToVec::new(),
            ExtractPosition::new(),
            Average::new(),
            CircleBuoy::new(context, DEPTH, 0.0, lateral_power),
        )
    )
}
