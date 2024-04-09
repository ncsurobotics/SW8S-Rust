use crate::comms::control_board::ControlBoard;
use crate::vision::Offset2D;
use crate::vision::RelPos;
use crate::vision::RelPosAngle;

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use derive_getters::Getters;
use num_traits::clamp;
use num_traits::Pow;
use std::ops::Rem;
use std::time::Duration;
use tokio::time::sleep;

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
            println!("Modify value: {:#?}", input);
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
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for AdjustMovement<'_, T> {
    async fn execute(&mut self) -> Result<()> {
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
            println!("Modify value: {:#?}", input);
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

impl<T, V> ActionMod<Result<V>> for AdjustMovementAngle<'_, T>
where
    V: RelPos<Number = f64> + Sync + Send + Debug,
{
    fn modify(&mut self, input: &Result<V>) {
        const MIN_TO_CHANGE_ANGLE: f32 = 0.1;
        const ANGLE_DIFF: f32 = 7.0;

        if let Ok(input) = input {
            println!("Modify value: {:#?}", input);
            if !input.offset().x().is_nan() && !input.offset().y().is_nan() {
                self.x = *input.offset().x() as f32;
                self.yaw_adjust -= if self.x.abs() > MIN_TO_CHANGE_ANGLE {
                    self.x * ANGLE_DIFF
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
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for AdjustMovementAngle<'_, T>
{
    async fn execute(&mut self) -> Result<()> {
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
        println!(
            "Current Yaw: {:#?}",
            self.context
                .get_control_board()
                .responses()
                .get_angles()
                .await
                .map(|angles| *angles.yaw())
        );
        println!("Adjusted Yaw: {}", yaw);

        println!("Prior x: {}", self.x);
        let mut x = self.x.signum() * self.x.abs().pow(ADJUST_VAL);
        if x.abs() < MIN_X {
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
            println!("Modify value: {:#?}", input);
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
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for CenterMovement<'_, T> {
    async fn execute(&mut self) -> Result<()> {
        const FACTOR: f32 = 1.5;
        const MIN_SPEED: f32 = 0.3;
        #[allow(dead_code)]
        const MIN_YAW: f32 = 5.0;

        let mut x = self.x.pow(FACTOR);
        let mut y = self.y.pow(FACTOR);
        let yaw = self.yaw;

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

/// Specifies replacement or adjustment (+ value)
#[derive(Debug, Clone)]
pub enum AdjustType<T> {
    Replace(T),
    Adjust(T),
}

/// Modification for a stability assist 2 command
///
/// When values are None, they do not cause adjustments
#[derive(Debug, Clone, Default, Getters)]
pub struct Stability2Adjust {
    x: Option<AdjustType<f32>>,
    y: Option<AdjustType<f32>>,
    target_pitch: Option<AdjustType<f32>>,
    target_roll: Option<AdjustType<f32>>,
    target_yaw: Option<AdjustType<f32>>,
    target_depth: Option<AdjustType<f32>>,
}

impl Stability2Adjust {
    pub const fn const_default() -> Self {
        Self {
            x: None,
            y: None,
            target_pitch: None,
            target_roll: None,
            target_yaw: None,
            target_depth: None,
        }
    }

    /// Convert all the invalid IEEE states into None
    fn address_ieee(val: AdjustType<f32>) -> Option<AdjustType<f32>> {
        match val {
            AdjustType::Replace(val) | AdjustType::Adjust(val)
                if val.is_nan() | val.is_infinite() | val.is_subnormal() =>
            {
                None
            }
            val => Some(val),
        }
    }

    /// Bounds speeds to [-1, 1]
    fn bound_speed(val: Option<AdjustType<f32>>) -> Option<AdjustType<f32>> {
        const MIN_SPEED: f32 = -1.0;
        const MAX_SPEED: f32 = 1.0;

        val.map(|val| match val {
            AdjustType::Replace(val) => AdjustType::Replace(clamp(val, MIN_SPEED, MAX_SPEED)),
            AdjustType::Adjust(val) => AdjustType::Adjust(val),
        })
    }

    /// Bounds rotations to 360 degrees
    fn bound_rot(val: Option<AdjustType<f32>>) -> Option<AdjustType<f32>> {
        const MAX_DEGREES: f32 = 360.0;

        val.map(|val| match val {
            AdjustType::Replace(val) => AdjustType::Replace(val.rem(MAX_DEGREES)),
            AdjustType::Adjust(val) => AdjustType::Adjust(val),
        })
    }

    pub fn set_x(&mut self, x: AdjustType<f32>) -> &Self {
        self.x = Self::bound_speed(Self::address_ieee(x));
        self
    }

    pub fn set_y(&mut self, y: AdjustType<f32>) -> &Self {
        self.y = Self::bound_speed(Self::address_ieee(y));
        self
    }

    pub fn set_target_pitch(&mut self, target_pitch: AdjustType<f32>) -> &Self {
        self.target_pitch = Self::bound_rot(Self::address_ieee(target_pitch));
        self
    }

    pub fn set_target_roll(&mut self, target_roll: AdjustType<f32>) -> &Self {
        self.target_roll = Self::bound_rot(Self::address_ieee(target_roll));
        self
    }

    pub fn set_target_yaw(&mut self, target_yaw: AdjustType<f32>) -> &Self {
        self.target_yaw = Self::bound_rot(Self::address_ieee(target_yaw));
        self
    }

    pub fn set_target_depth(&mut self, target_depth: AdjustType<f32>) -> &Self {
        self.target_depth = Self::bound_rot(Self::address_ieee(target_depth));
        self
    }
}

/// Stores the command to send to stability assist 2
///
/// If target_yaw is None, it is set to the current yaw on first execution
#[derive(Debug, Clone)]
pub struct Stability2Pos {
    x: f32,
    y: f32,
    target_pitch: f32,
    target_roll: f32,
    target_yaw: Option<f32>, // set to current if uninitialized
    target_depth: f32,
}

impl Stability2Pos {
    pub const fn new(
        x: f32,
        y: f32,
        target_pitch: f32,
        target_roll: f32,
        target_yaw: Option<f32>,
        target_depth: f32,
    ) -> Self {
        Self {
            x,
            y,
            target_pitch,
            target_roll,
            target_yaw,
            target_depth,
        }
    }

    /// Executes the position in stability assist
    pub async fn exec(&mut self, board: &ControlBoard<WriteHalf<SerialStream>>) -> Result<()> {
        const SLEEP_LEN: Duration = Duration::from_millis(100);

        // Intializes yaw to current value
        if self.target_yaw.is_none() {
            // Repeats until an angle measurement exists
            loop {
                if let Some(angles) = board.responses().get_angles().await {
                    self.target_yaw = Some(*angles.yaw());
                    break;
                }
                sleep(SLEEP_LEN).await;
            }
        }

        println!("Stability 2 speed set: {:#?}", self);

        board
            .stability_2_speed_set(
                self.x,
                self.y,
                self.target_pitch,
                self.target_roll,
                self.target_yaw.unwrap(),
                self.target_depth,
            )
            .await
    }

    /// Sets speed, bounded to [-1, 1]
    fn set_speed(base: f32, adjuster: Option<AdjustType<f32>>) -> f32 {
        const MIN_SPEED: f32 = -1.0;
        const MAX_SPEED: f32 = 1.0;

        adjuster
            .map(|val| match val {
                AdjustType::Replace(val) => val,
                AdjustType::Adjust(val) => clamp(base + val, MIN_SPEED, MAX_SPEED),
            })
            .unwrap_or(base)
    }

    /// Set rotation, bounded to 360 degrees
    fn set_rot(base: f32, adjuster: Option<AdjustType<f32>>) -> f32 {
        const MAX_DEGREES: f32 = 360.0;

        adjuster
            .map(|val| match val {
                AdjustType::Replace(val) => val,
                AdjustType::Adjust(val) => (val + base).rem(MAX_DEGREES),
            })
            .unwrap_or(base)
    }

    /// Adjusts the position according to `adjuster`.
    ///
    /// The x and y fields are bounded to [-1, 1].
    /// The pitch, roll, yaw, depth fields wrap around 360 degrees.
    pub fn adjust(&mut self, adjuster: &Stability2Adjust) -> &Self {
        println!("Stability 2 pre-adjust: {:#?}", self);
        println!("Adjuster: {:#?}", adjuster);

        self.x = Self::set_speed(self.x, adjuster.x().clone());
        self.y = Self::set_speed(self.y, adjuster.y().clone());

        self.target_pitch = Self::set_rot(self.target_pitch, adjuster.target_pitch().clone());
        self.target_roll = Self::set_rot(self.target_roll, adjuster.target_roll().clone());
        self.target_depth = Self::set_rot(self.target_depth, adjuster.target_depth().clone());

        // Accounting for uninitialized yaw
        self.target_yaw = if let Some(target_yaw) = self.target_yaw {
            Some(Self::set_rot(target_yaw, adjuster.target_yaw().clone()))
        } else if let Some(AdjustType::Replace(adjuster_yaw)) = adjuster.target_yaw() {
            Some(*adjuster_yaw)
        } else {
            None
        };

        println!("Stability 2 post-adjust: {:#?}", self);
        self
    }

    pub const fn const_default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, None, 0.0)
    }
}

impl Default for Stability2Pos {
    fn default() -> Self {
        Self::const_default()
    }
}

#[derive(Debug)]
pub struct Stability2Movement<'a, T> {
    context: &'a T,
    pose: Stability2Pos,
}

impl<T> Action for Stability2Movement<'_, T> {}

impl<'a, T> Stability2Movement<'a, T> {
    pub const fn new(context: &'a T, pose: Stability2Pos) -> Self {
        Self { context, pose }
    }

    pub fn uninitialized(context: &'a T) -> Self {
        Self {
            context,
            pose: Stability2Pos::default(),
        }
    }
}

impl<T> ActionMod<Stability2Pos> for Stability2Movement<'_, T> {
    fn modify(&mut self, input: &Stability2Pos) {
        self.pose = input.clone();
    }
}

impl<T> ActionMod<Stability2Adjust> for Stability2Movement<'_, T> {
    fn modify(&mut self, input: &Stability2Adjust) {
        self.pose.adjust(input);
    }
}

#[async_trait]
impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for Stability2Movement<'a, T>
{
    async fn execute(&mut self) -> Result<()> {
        self.pose.exec(self.context.get_control_board()).await
    }
}

#[async_trait]
impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<()> for Stability2Movement<'a, T> {
    async fn execute(&mut self) -> () {
        let _ = self.pose.exec(self.context.get_control_board()).await;
    }
}

/// Generates a yaw adjustment from an x axis set, multiplying by angle_diff
///
/// Does not set a yaw adjustment if the x difference is below 0.1
pub fn linear_yaw_from_x(mut input: Stability2Adjust, angle_diff: f32) -> Stability2Adjust {
    const MIN_TO_CHANGE_ANGLE: f32 = 0.1;
    if let Some(AdjustType::Replace(x)) = input.x() {
        if *x > MIN_TO_CHANGE_ANGLE {
            input.set_target_yaw(AdjustType::Adjust(x * angle_diff));
        };
    };
    input
}

/// [`linear_yaw_from_x`] with a default value
pub const fn default_linear_yaw_from_x() -> fn(Stability2Adjust) -> Stability2Adjust {
    const ANGLE_DIFF: f32 = 7.0;
    |input| linear_yaw_from_x(input, ANGLE_DIFF)
}

/// Action version of [`linear_yaw_from_x`]
#[derive(Debug)]
pub struct LinearYawFromX<T> {
    angle_diff: f32,
    pose: T,
}

impl<T> Action for LinearYawFromX<T> {}

impl LinearYawFromX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(angle_diff: f32) -> Self {
        Self {
            angle_diff,
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> LinearYawFromX<T> {
    pub fn new(angle_diff: f32) -> Self {
        Self {
            angle_diff,
            pose: T::default(),
        }
    }
}

impl<T: Default> Default for LinearYawFromX<T> {
    fn default() -> Self {
        const ANGLE_DIFF: f32 = 7.0;
        Self::new(ANGLE_DIFF)
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for LinearYawFromX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

#[async_trait]
impl ActionExec<Stability2Adjust> for LinearYawFromX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        linear_yaw_from_x(self.pose.clone(), self.angle_diff)
    }
}

#[async_trait]
impl ActionExec<Stability2Adjust> for LinearYawFromX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        linear_yaw_from_x(self.pose.clone(), self.angle_diff)
    }
}

#[derive(Debug)]
pub struct StripY<T> {
    pose: T,
}

impl<T> Action for StripY<T> {}

impl StripY<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> StripY<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for StripY<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for StripY<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

#[async_trait]
impl ActionExec<Stability2Adjust> for StripY<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut pose = self.pose.clone();
        pose.y = None;
        pose
    }
}

#[async_trait]
impl ActionExec<Stability2Adjust> for StripY<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.y = None;
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct OffsetToPose<T> {
    offset: T,
}

impl<T> Action for OffsetToPose<T> {}

impl<T> OffsetToPose<T> {
    pub const fn new(offset: T) -> Self {
        Self { offset }
    }
}

impl<T: Default> Default for OffsetToPose<T> {
    fn default() -> Self {
        Self {
            offset: T::default(),
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<T> for OffsetToPose<T> {
    fn modify(&mut self, input: &T) {
        self.offset = input.clone();
    }
}

impl<T: Send + Sync + Clone + Default> ActionMod<Option<T>> for OffsetToPose<T> {
    fn modify(&mut self, input: &Option<T>) {
        if let Some(input) = input {
            self.offset = input.clone();
        } else {
            self.offset = T::default();
        }
    }
}

impl<T: Send + Sync + Clone + Default> ActionMod<anyhow::Result<T>> for OffsetToPose<T> {
    fn modify(&mut self, input: &anyhow::Result<T>) {
        if let Ok(input) = input {
            self.offset = input.clone();
        } else {
            self.offset = T::default();
        }
    }
}

#[async_trait]
impl ActionExec<Stability2Adjust> for OffsetToPose<Offset2D<f64>> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut adjust = Stability2Adjust::default();
        adjust.set_x(AdjustType::Replace(*self.offset.x() as f32));
        adjust.set_y(AdjustType::Replace(*self.offset.y() as f32));
        adjust
    }
}
