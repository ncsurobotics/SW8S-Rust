use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;

use super::{action::Action, action_context::GetMainElectronicsBoard};

#[derive(Debug)]
pub struct WaitArm<'a, T> {
    context: &'a T,
}

impl<'a, T> WaitArm<'a, T> {
    pub const fn new(context: &'a T) -> Self {
        Self { context }
    }
}

#[async_trait]
impl<T: GetMainElectronicsBoard> Action<()> for WaitArm<'_, T> {
    /// Wait for system to be armed
    async fn execute(self) {
        while !self
            .context
            .get_main_electronics_board()
            .thruster_arm()
            .await
            .unwrap_or(false)
        {
            sleep(Duration::from_millis(10)).await;
        }
    }
}
