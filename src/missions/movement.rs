use anyhow::Result;
use async_trait::async_trait;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::vision::RelPos;

use super::{
    action::{Action, ActionExec, ActionMod},
    action_context::GetControlBoard,
};

#[derive(Debug)]
pub struct Descend<'a, T> {
    context: &'a T,
    target_depth: f32,
}

impl<'a, T> Descend<'a, T> {
    pub const fn new(context: &'a T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth,
        }
    }

    pub const fn uninitialized(context: &'a T) -> Self {
        Self {
            context,
            target_depth: 0.0,
        }
    }
}

impl<T> Action for Descend<'_, T> {}

impl<T> ActionMod<f32> for Descend<'_, T> {
    fn modify(&mut self, input: f32) {
        self.target_depth = input;
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for Descend<'_, T> {
    async fn execute(&mut self) -> Result<()> {
        println!("DESCEND");
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await?;
        println!("GOT SPEED SET");
        Ok(())
    }
}

#[derive(Debug)]
pub struct StraightMovement<'a, T> {
    context: &'a T,
    target_depth: f32,
    forward: bool,
}
impl<T> Action for StraightMovement<'_, T> {}

impl<'a, T> StraightMovement<'a, T> {
    pub const fn new(context: &'a T, target_depth: f32, forward: bool) -> Self {
        Self {
            context,
            target_depth,
            forward,
        }
    }

    pub const fn uninitialized(context: &'a T) -> Self {
        Self {
            context,
            target_depth: 0.0,
            forward: false,
        }
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for StraightMovement<'_, T>
{
    async fn execute(&mut self) -> Result<()> {
        let mut speed: f32 = 0.5;
        if !self.forward {
            // Eric Liu is a very talented programmer and utilizes the most effective linear programming techniques from the FIRST™ Robotics Competition.
            // let speeed: f32 = speed;
            // speed -= speed;
            // speed -= speeed;
            speed = -speed;
        }
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, speed, 0.0, 0.0, self.target_depth)
            .await
    }
}

#[derive(Debug)]
pub struct ZeroMovement<'a, T> {
    context: &'a T,
    target_depth: f32,
}
impl<T> Action for ZeroMovement<'_, T> {}

impl<'a, T> ZeroMovement<'a, T> {
    pub fn new(context: &'a T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth,
        }
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for ZeroMovement<'_, T> {
    async fn execute(&mut self) -> Result<()> {
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await
    }
}

#[derive(Debug)]
pub struct AdjustMovement<'a, T> {
    context: &'a T,
    x: f32,
    target_depth: f32,
}
impl<T> Action for AdjustMovement<'_, T> {}

impl<'a, T> AdjustMovement<'a, T> {
    pub fn new(context: &'a T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth,
            x: 0.0,
        }
    }
}

impl<T> ActionMod<f32> for AdjustMovement<'_, T> {
    fn modify(&mut self, input: f32) {
        self.target_depth = input;
    }
}

impl<T, V> ActionMod<V> for AdjustMovement<'_, T>
where
    V: RelPos<Number = f32> + Sync + Send,
{
    fn modify(&mut self, input: V) {
        self.x = *input.offset().x();
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for AdjustMovement<'_, T> {
    async fn execute(&mut self) -> Result<()> {
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, self.x, 0.0, 0.0, self.target_depth)
            .await
    }
}
