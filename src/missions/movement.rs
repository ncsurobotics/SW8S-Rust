use crate::vision::RelPos;
use anyhow::anyhow;
use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

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
    fn modify(&mut self, input: &f32) {
        self.target_depth = *input;
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for Descend<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
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
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for StraightMovement<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
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
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for ZeroMovement<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
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
    fn modify(&mut self, input: &f32) {
        self.target_depth = *input;
    }
}

impl<T, V> ActionMod<Result<V>> for AdjustMovement<'_, T>
where
    V: RelPos<Number = f64> + Sync + Send + Debug,
{
    fn modify(&mut self, input: &Result<V>) {
        if let Ok(input) = input {
            println!("Modify value: {:?}", input);
            if !input.offset().x().is_nan() || !input.offset().y().is_nan() {
                self.x = *input.offset().x() as f32;
            } else {
                self.x = 0.0;
            }
        } else {
            self.x = 0.0;
        }
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for AdjustMovement<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(self.x, 0.5, 0.0, 0.0, self.target_depth)
            .await
    }
}

#[derive(Debug)]
pub struct CountTrue {
    target: u32,
    count: u32,
}

impl CountTrue {
    pub fn new(target: u32) -> Self {
        CountTrue { target, count: 0 }
    }
}

impl Action for CountTrue {}

impl<T: Send + Sync> ActionMod<Result<T>> for CountTrue {
    fn modify(&mut self, input: &Result<T>) {
        if input.is_ok() {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
    }
}

#[async_trait]
impl ActionExec for CountTrue {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        if self.count == self.target {
            Ok(())
        } else {
            Err(anyhow!("Under count"))
        }
    }
}

#[derive(Debug)]
pub struct CountFalse {
    target: u32,
    count: u32,
}

impl CountFalse {
    pub fn new(target: u32) -> Self {
        CountFalse { target, count: 0 }
    }
}

impl Action for CountFalse {}

impl<T: Send + Sync> ActionMod<Result<T>> for CountFalse {
    fn modify(&mut self, input: &Result<T>) {
        if input.is_err() {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
    }
}

#[async_trait]
impl ActionExec for CountFalse {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        if self.count == self.target {
            Ok(())
        } else {
            Err(anyhow!("Under count"))
        }
    }
}
