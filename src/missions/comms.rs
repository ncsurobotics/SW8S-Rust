use std::marker::PhantomData;

use anyhow::Result;
use async_trait::async_trait;
use tokio::io::{AsyncWrite, WriteHalf};

use crate::comms::stubborn_serial::StubbornSerialStream;

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

#[async_trait]
impl<T: GetControlBoard<WriteHalf<StubbornSerialStream>>> ActionExec for StartBno055<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        self.context
            .get_control_board()
            .bno055_periodic_read(true)
            .await
    }
}
