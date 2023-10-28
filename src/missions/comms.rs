use anyhow::Result;
use async_trait::async_trait;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use super::{
    action::{Action, ActionExec},
    action_context::GetControlBoard,
};

#[derive(Debug)]
pub struct StartBno055<T> {
    context: T,
}

impl<T> StartBno055<T> {
    pub const fn new(context: T) -> Self {
        Self { context }
    }
}

impl<T> Action for StartBno055<T> {}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for StartBno055<T> {
    async fn execute(&mut self) -> Result<()> {
        self.context
            .get_control_board()
            .bno055_periodic_read(true)
            .await
    }
}
