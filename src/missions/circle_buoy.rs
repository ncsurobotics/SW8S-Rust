use crate::{
    act_nest,
    missions::{
        action::ActionChain,
        extra::ToVec,
        vision::{Average, ExtractPosition, VisionNorm},
    },
    vision::{
        path::{Path, Yuv},
        Offset2D,
    },
};

use super::{
    action::{Action, ActionExec, ActionMod, ActionSequence},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    movement::ZeroMovement,
};

use async_trait::async_trait;

use opencv::core::Size;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

pub struct CircleBuoy<'a, T> {
    context: &'a T,
    target_depth: f32,
    lateral_power: f32,
    rotation: f32,
}

impl<'a, T> CircleBuoy<'a, T> {
    pub fn new(context: &'a T, target_depth: f32, _forward_power: f32, lateral_power: f32) -> Self {
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
    const DEPTH: f32 = -0.5;

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
            VisionNorm::<Con, Path, f64>::new(
                context,
                Path::new(
                    (Yuv {
                        y: 128,
                        u: 0,
                        v: 127
                    })..=(Yuv {
                        y: 255,
                        u: 100,
                        v: 255,
                    }),
                    20.0..=800.0,
                    4,
                    Size::from((400, 300)),
                    3,
                )
            ),
            ToVec::new(),
            ExtractPosition::new(),
            Average::new(),
            CircleBuoy::new(context, DEPTH, 0.0, lateral_power),
        )
    )
}
