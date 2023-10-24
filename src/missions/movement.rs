use anyhow::Result;
use async_trait::async_trait;
use tokio_serial::SerialStream;

use super::{
    action::{Action, ActionExec, ActionMod},
    action_context::GetControlBoard,
};

#[derive(Debug)]
pub struct Descend<T> {
    context: T,
    target_depth: f32,
}

impl<T> Descend<T> {
    pub const fn new(context: T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth,
        }
    }

    pub const fn uninitialized(context: T) -> Self {
        Self {
            context,
            target_depth: 0.0,
        }
    }
}

impl<T> Action for Descend<T> {}

impl<T> ActionMod<f32> for Descend<T> {
    fn modify(&mut self, input: f32) {
        self.target_depth = input;
    }
}

#[async_trait]
impl<T: GetControlBoard<SerialStream>> ActionExec<Result<()>> for Descend<T> {
    async fn execute(&mut self) -> Result<()> {
        self.context
            .get_control_board()
            .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await
    }
}
