use crate::comms::control_board::ControlBoard;
use crate::vision::RelPos;
use crate::vision::RelPosAngle;
use anyhow::anyhow;
use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use num_traits::Pow;
use opencv::core::abs;
use tokio::io::AsyncWrite;
use tokio::io::WriteHalf;
use tokio::sync::OnceCell;
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
pub struct AdjustMovementAngle<'a, T> {
    context: &'a T,
    x: f32,
    yaw_adjust: f32,
    target_depth: f32,
}
impl<T> Action for AdjustMovementAngle<'_, T> {}

impl<'a, T> AdjustMovementAngle<'a, T> {
    pub fn new(context: &'a T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth,
            x: 0.0,
            yaw_adjust: 0.0,
        }
    }
}

impl<T> ActionMod<f32> for AdjustMovementAngle<'_, T> {
    fn modify(&mut self, input: &f32) {
        self.target_depth = *input;
    }
}

/*
impl<T, V> ActionMod<Result<V>> for AdjustMovementAngle<'_, T>
where
    V: RelPosAngle<Number = f64> + Sync + Send + Debug,
{
    fn modify(&mut self, input: &Result<V>) {
        if let Ok(input) = input {
            println!("Modify value: {:?}", input);
            if !input.offset().x().is_nan() && !input.offset().y().is_nan() {
                self.x = *input.offset().x() as f32;
                self.yaw = *input.offset_angle().angle() as f32;
            } else {
                self.x = 0.0;
                self.yaw = 0.0;
            }
        } else {
            self.x = 0.0;
            self.yaw = 0.0;
        }
    }
}
*/

static ANGLE_BASE_VALUE: OnceCell<f32> = OnceCell::const_new();
async fn angle_base_value<T: AsyncWrite + Unpin>(board: &ControlBoard<T>) -> f32 {
    *ANGLE_BASE_VALUE
        .get_or_init(|| async {
            let mut angles = board.responses().get_angles().await;
            while angles.is_none() {
                angles = board.responses().get_angles().await;
            }
            *angles.unwrap().yaw()
        })
        .await
}

impl<T, V> ActionMod<Result<V>> for AdjustMovementAngle<'_, T>
where
    V: RelPos<Number = f64> + Sync + Send + Debug,
{
    fn modify(&mut self, input: &Result<V>) {
        const MIN_TO_CHANGE_ANGLE: f32 = 0.1;
        const ANGLE_DIFF: f32 = 3.0;

        if let Ok(input) = input {
            println!("Modify value: {:?}", input);
            if !input.offset().x().is_nan() && !input.offset().y().is_nan() {
                self.x = *input.offset().x() as f32;
                self.yaw_adjust += if self.x.abs() > MIN_TO_CHANGE_ANGLE {
                    (self.x / self.x.abs()) * ANGLE_DIFF
                } else {
                    0.0
                };
                println!("YAW ADJUST: {}", self.yaw_adjust);
            } else {
                self.x = 0.0;
            }
        } else {
            self.x = 0.0;
        }
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for AdjustMovementAngle<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        const ADJUST_VAL: f32 = 1.5;
        const MIN_X: f32 = 0.3;

        let yaw = if let Some(angles) = self.context.get_control_board().get_initial_angles().await
        {
            println!("Initial Yaw: {}", angles.yaw());
            let mut inner_yaw = angles.yaw() + self.yaw_adjust;
            if inner_yaw.abs() > 180.0 {
                let sign = inner_yaw / inner_yaw.abs();
                inner_yaw = -(inner_yaw - (sign * 180.0)); // TODO: confirm this math
            }
            inner_yaw
        } else {
            0.0
        };
        println!("Adjusted Yaw: {}", yaw);

        let mut x = self.x.pow(ADJUST_VAL);
        if x < MIN_X {
            x = 0.0;
        }
        println!("Setting x to {x}");

        self.context
            .get_control_board()
            .stability_2_speed_set(x, 0.5, 0.0, 0.0, yaw, self.target_depth)
            .await
    }
}

#[derive(Debug)]
pub struct CenterMovement<'a, T> {
    context: &'a T,
    x: f32,
    y: f32,
    yaw: f32,
    target_depth: f32,
}
impl<T> Action for CenterMovement<'_, T> {}

impl<'a, T> CenterMovement<'a, T> {
    pub fn new(context: &'a T, target_depth: f32) -> Self {
        Self {
            context,
            target_depth,
            x: 0.0,
            y: 0.0,
            yaw: 0.0,
        }
    }
}

impl<T> ActionMod<f32> for CenterMovement<'_, T> {
    fn modify(&mut self, input: &f32) {
        self.target_depth = *input;
    }
}

impl<T, V> ActionMod<Result<V>> for CenterMovement<'_, T>
where
    V: RelPosAngle<Number = f64> + Sync + Send + Debug,
{
    fn modify(&mut self, input: &Result<V>) {
        if let Ok(input) = input {
            println!("Modify value: {:?}", input);
            if !input.offset().x().is_nan() && !input.offset().y().is_nan() {
                self.x = *input.offset().x() as f32;
                self.y = *input.offset().y() as f32;
                self.yaw = *input.offset_angle().angle() as f32;
            } else {
                self.x = 0.0;
                self.y = 0.0;
            }
        } else {
            self.x = 0.0;
            self.y = 0.0;
        }
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for CenterMovement<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        const FACTOR: f32 = 1.5;
        const MIN_SPEED: f32 = 0.3;
        const MIN_YAW: f32 = 5.0;

        let mut x = self.x.pow(FACTOR);
        let mut y = self.y.pow(FACTOR);
        let mut yaw = self.yaw;

        if x < MIN_SPEED {
            x = 0.0;
        }
        if y < MIN_SPEED {
            y = 0.0;
        }

        self.context
            .get_control_board()
            .stability_2_speed_set(x, y, 0.0, 0.0, yaw, self.target_depth)
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
        println!("COUNTING TRUE: {} ? {}", self.count, self.target);
    }
}

#[async_trait]
impl ActionExec for CountTrue {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        println!("CHECKING TRUE: {} ? {}", self.count, self.target);
        if self.count < self.target {
            println!("Under count");
            Ok(())
        } else {
            Err(anyhow!("At count"))
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
        println!("COUNTING FALSE: {} ? {}", self.count, self.target);
    }
}

#[async_trait]
impl ActionExec for CountFalse {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        println!("CHECKING FALSE: {} ? {}", self.count, self.target);
        if self.count < self.target {
            println!("Under count");
            Ok(())
        } else {
            Err(anyhow!("At count"))
        }
    }
}
