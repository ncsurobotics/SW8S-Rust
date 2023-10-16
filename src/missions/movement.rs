use anyhow::Result;
use async_trait::async_trait;
use tokio_serial::SerialStream;

use super::{
    action::{Action, ActionMod},
    action_context::GetControlBoard,
};

#[derive(Debug)]
struct Descend<T> {
    context: T,
    target_depth: f32,
}

impl<T> Descend<T> {
    const fn new(context: T) -> Self {
        Self {
            context,
            target_depth: 0.0,
        }
    }
}

impl<T> ActionMod<f32> for Descend<T> {
    fn modify(&mut self, input: f32) {
        self.target_depth = input;
    }
}

#[async_trait]
impl<T: GetControlBoard<SerialStream>> Action<Result<()>> for Descend<T> {
    async fn execute(self) -> Result<()> {
        self.context
            .get_control_board()
            .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await
    }
}
