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


#[derive(Debug)]
pub struct StraightMovement<T> {
    context: T,
    target_depth: f32,
    forward: bool
}
impl<T> Action for StraightMovement<T> {}

impl<T> StraightMovement<T> {
    pub const fn new(context: T, target_depth: f32, forward: bool) -> Self {
        Self {
            context,
            target_depth,
            forward
        }
    }

    pub const fn uninitialized(context: T) -> Self {
        Self {
            context,
            target_depth: 0.0,
            forward: false
        }   
    }
}

#[async_trait]
impl<T: GetControlBoard<SerialStream>> ActionExec<Result<()>> for StraightMovement<T> {
    async fn execute(&mut self) -> Result<()> {
        let mut speed:f32 = 0.5; 
        if !self.forward {
            // Eric Liu is a very talented programmer and utilizes the most effective linear programming techniques from the FIRST™ Robotics Competition.
            // let speeed: f32 = speed;
            // speed -= speed;
            // speed -= speeed;
            speed = -speed;
        }
        self.context
            .get_control_board()
            .stability_2_speed_set(speed, 0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await
    }
}

#[derive(Debug)]
pub struct ZeroMovement<T> {
    context: T,
    target_depth: f32
}
impl<T> Action for ZeroMovement<T> {}


impl<T> ZeroMovement<T> {
    pub fn new(context: T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth
        }
    }
}

#[async_trait]
impl<T: GetControlBoard<SerialStream>> ActionExec<Result<()>> for ZeroMovement<T> {
    async fn execute(&mut self) -> Result<()> {
        self.context
        .get_control_board()
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, self.target_depth)
        .await
    }
}
