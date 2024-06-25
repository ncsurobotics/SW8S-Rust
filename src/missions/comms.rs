use anyhow::Result;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use super::{
    action::{Action, ActionExec},
    action_context::GetControlBoard,
};

#[derive(Debug)]
pub struct StartBno055<'a, T> {
    context: &'a T,
}

impl<'a, T> StartBno055<'a, T> {
    pub const fn new(context: &'a T) -> Self {
        Self { context }
    }
}

impl<T> Action for StartBno055<'_, T> {}

impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for StartBno055<'_, T> {
    async fn execute(&mut self) -> Result<()> {
        self.context
            .get_control_board()
            .bno055_periodic_read(true)
            .await
    }
}
