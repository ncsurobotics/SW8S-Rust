use crate::comms::control_board::ControlBoard;
use crate::comms::control_board::LAST_YAW;
use crate::logln;
use crate::vision::DrawRect2d;
use crate::vision::Offset2D;
use crate::vision::RelPos;
use crate::vision::RelPosAngle;

use anyhow::Result;
use core::fmt::Debug;
use derive_getters::Getters;
use num_traits::abs;
use num_traits::clamp;
use num_traits::Pow;
use num_traits::Zero;
use std::marker::PhantomData;
use std::ops::Rem;
use std::sync::Mutex;
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

impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>> for Descend<'_, T> {
    async fn execute(&mut self) -> Result<()> {
        logln!("DESCEND");
        const SLEEP_LEN: Duration = Duration::from_millis(100);

        let cntrl = self.context.get_control_board();

        let cur_yaw;

        // Intializes yaw to current value
        // Repeats until an angle measurement exists
        loop {
            if let Some(angles) = cntrl.get_initial_angles().await {
                cur_yaw = *angles.yaw();
                break;
            } else {
                cntrl.bno055_periodic_read(true).await?;
            }
            sleep(SLEEP_LEN).await;
        }

        cntrl
            .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, cur_yaw, self.target_depth)
            .await?;
        logln!("GOT SPEED SET");
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

impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for StraightMovement<'_, T>
{
    async fn execute(&mut self) -> Result<()> {
        let mut speed: f32 = 0.6;
        if !self.forward {
            // Eric Liu is a very talented programmer and utilizes the most effective linear programming techniques from the FIRST™ Robotics Competition.
            // let speeed: f32 = speed;
            // speed -= speed;
            // speed -= speeed;
            speed = -speed;
        }

        let cntrl_board = self.context.get_control_board();
        let mut cur_angles = cntrl_board.responses().get_angles().await;
        while cur_angles.is_none() {
            cur_angles = cntrl_board.responses().get_angles().await;
        }

        cntrl_board
            .stability_2_speed_set(
                0.0,
                speed,
                0.0,
                0.0,
                *cur_angles.unwrap().yaw(),
                self.target_depth,
            )
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
            logln!("Modify value: {:#?}", input);
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
            logln!("Modify value: {:#?}", input);
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
        const ANGLE_DIFF: f32 = 20.0;

        if let Ok(input) = input {
            logln!("Modify value: {:#?}", input);
            if !input.offset().x().is_nan() && !input.offset().y().is_nan() {
                self.x = *input.offset().x() as f32;
                self.yaw_adjust += if self.x.abs() > MIN_TO_CHANGE_ANGLE {
                    self.x * ANGLE_DIFF
                } else {
                    0.0
                };
                logln!("YAW ADJUST: {}", self.yaw_adjust);
            } else {
                self.x = 0.0;
            }
        } else {
            self.x = 0.0;
        }
    }
}

impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for AdjustMovementAngle<'_, T>
{
    #[allow(clippy::await_holding_lock)]
    async fn execute(&mut self) -> Result<()> {
        const ADJUST_VAL: f32 = 1.5;
        const MIN_X: f32 = 0.3;

        let yaw = if let Some(angles) = self.context.get_control_board().get_initial_angles().await
        {
            logln!("Initial Yaw: {}", angles.yaw());
            let mut inner_yaw = angles.yaw() + self.yaw_adjust;
            if inner_yaw.abs() > 180.0 {
                let sign = inner_yaw / inner_yaw.abs();
                inner_yaw = -(inner_yaw - (sign * 180.0)); // TODO: confirm this math
            }
            inner_yaw
        } else {
            0.0
        };
        logln!(
            "Current Yaw: {:#?}",
            self.context
                .get_control_board()
                .responses()
                .get_angles()
                .await
                .map(|angles| *angles.yaw())
        );
        logln!("Adjusted Yaw: {}", yaw);

        logln!("Prior x: {}", self.x);
        let mut x = self.x.signum() * self.x.abs().pow(ADJUST_VAL);
        if x.abs() < MIN_X {
            x = 0.0;
        }
        logln!("Setting x to {x}");

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
            logln!("Modify value: {:#?}", input);
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
        #[allow(clippy::await_holding_lock)]
        if self.target_yaw.is_none() {
            let last_yaw = LAST_YAW.lock().unwrap();
            if let Some(last_yaw) = *last_yaw {
                self.target_yaw = Some(last_yaw);
            } else {
                drop(last_yaw);
                // Repeats until an angle measurement exists
                loop {
                    if let Some(angles) = board.responses().get_angles().await {
                        self.target_yaw = Some(*angles.yaw());
                        break;
                    }
                    sleep(SLEEP_LEN).await;
                }
            }
        }

        //logln!("Stability 2 speed set: {:#?}", self);

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
        //logln!("Stability 2 pre-adjust: {:#?}", self);
        //logln!("Adjuster: {:#?}", adjuster);

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

        logln!("Stability 2 post-adjust: {:#?}", self);
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

impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for Stability2Movement<'a, T>
{
    async fn execute(&mut self) -> Result<()> {
        self.pose.exec(self.context.get_control_board()).await
    }
}

impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<()> for Stability2Movement<'a, T> {
    async fn execute(&mut self) {
        let _ = self.pose.exec(self.context.get_control_board()).await;
    }
}

/// Generates a yaw adjustment from an x axis set, multiplying by angle_diff
///
/// Does not set a yaw adjustment if the x difference is below 0.1
pub fn linear_yaw_from_x(mut input: Stability2Adjust, angle_diff: f32) -> Stability2Adjust {
    const MIN_TO_CHANGE_ANGLE: f32 = 0.1;
    if let Some(AdjustType::Replace(x)) = input.x() {
        if abs(*x) > MIN_TO_CHANGE_ANGLE {
            input.set_target_yaw(AdjustType::Adjust(-x * angle_diff));
        };
    };
    input
}

/// Generates a yaw adjustment from an x axis set, multiplying by angle_diff
///
/// Does not set a yaw adjustment if the x difference is below 0.1
pub fn linear_yaw_from_x_stab1(mut input: Stability1Adjust, angle_diff: f32) -> Stability1Adjust {
    const MIN_TO_CHANGE_ANGLE: f32 = 0.1;
    if let Some(AdjustType::Replace(x)) = input.x() {
        if abs(*x) > MIN_TO_CHANGE_ANGLE {
            input.set_yaw_speed(AdjustType::Adjust(-x * angle_diff));
        };
    };
    input
}

pub fn aggressive_yaw_from_x(mut input: Stability2Adjust, angle_diff: f32) -> Stability2Adjust {
    if let Some(AdjustType::Replace(x)) = input.x() {
        if !x.is_zero() {
            input.set_target_yaw(AdjustType::Adjust(x.signum() * angle_diff));
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

impl LinearYawFromX<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
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

impl ActionExec<Stability2Adjust> for LinearYawFromX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        linear_yaw_from_x(self.pose.clone(), self.angle_diff)
    }
}

impl ActionExec<Stability2Adjust> for LinearYawFromX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        linear_yaw_from_x(self.pose.clone(), self.angle_diff)
    }
}

impl ActionExec<Stability1Adjust> for LinearYawFromX<&Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        linear_yaw_from_x_stab1(self.pose.clone(), self.angle_diff)
    }
}

impl ActionExec<Stability1Adjust> for LinearYawFromX<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        linear_yaw_from_x_stab1(self.pose.clone(), self.angle_diff)
    }
}

#[derive(Debug)]
pub struct FlipYaw<T> {
    pose: T,
}

impl<T> Action for FlipYaw<T> {}

impl FlipYaw<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl FlipYaw<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> FlipYaw<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for FlipYaw<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for FlipYaw<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for FlipYaw<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        logln!("YAW BEFORE: {:?}", self.pose.target_yaw);
        if let Some(ref mut yaw) = self.pose.target_yaw {
            match yaw {
                AdjustType::Adjust(ref mut y) => *y = -*y,
                AdjustType::Replace(ref mut y) => *y = -*y,
            }
        };
        logln!("YAW AFTER: {:?}", self.pose.target_yaw);
        self.pose.clone()
    }
}

impl ActionExec<Stability1Adjust> for FlipYaw<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        logln!("YAW BEFORE: {:?}", self.pose.yaw_speed);
        if let Some(ref mut yaw) = self.pose.yaw_speed {
            match yaw {
                AdjustType::Adjust(ref mut y) => *y = -*y,
                AdjustType::Replace(ref mut y) => *y = -*y,
            }
        };
        logln!("YAW AFTER: {:?}", self.pose.yaw_speed);
        self.pose.clone()
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

impl ActionExec<Stability2Adjust> for StripY<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.y = None;
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct ConfidenceY<T> {
    pose: T,
}

impl<T> Action for ConfidenceY<T> {}

impl ConfidenceY<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> ConfidenceY<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for ConfidenceY<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for ConfidenceY<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for ConfidenceY<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.y = Some(if let Some(AdjustType::Replace(x)) = self.pose.x {
            if x.is_zero() {
                AdjustType::Replace(0.2)
            } else {
                AdjustType::Adjust(0.1)
            }
        } else {
            AdjustType::Replace(0.2)
        });
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct SetY<T> {
    pose: T,
    y: AdjustType<f32>,
}

impl<T> Action for SetY<T> {}

impl SetY<Stability2Adjust> {
    pub const fn new(y: AdjustType<f32>) -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
            y,
        }
    }
}

impl SetY<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(y: AdjustType<f32>) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            y,
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for SetY<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for SetY<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.y = Some(self.y.clone());
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct FlipX<T> {
    pose: T,
}

impl<T> Action for FlipX<T> {}

impl FlipX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl FlipX<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> FlipX<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for FlipX<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for FlipX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for FlipX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(ref mut x) = self.pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(-x),
                AdjustType::Replace(x) => AdjustType::Replace(-x),
            };
        }
        self.pose.clone()
    }
}

impl ActionExec<Stability1Adjust> for FlipX<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        if let Some(ref mut x) = self.pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(-x),
                AdjustType::Replace(x) => AdjustType::Replace(-x),
            };
        }
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct FlipY<T> {
    pose: T,
}

impl<T> Action for FlipY<T> {}

impl FlipY<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl FlipY<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> FlipY<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for FlipY<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for FlipY<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for FlipY<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(ref mut y) = self.pose.y {
            *y = match *y {
                AdjustType::Adjust(y) => AdjustType::Adjust(-y),
                AdjustType::Replace(y) => AdjustType::Replace(-y),
            };
        }
        self.pose.clone()
    }
}

impl ActionExec<Stability1Adjust> for FlipY<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        if let Some(ref mut y) = self.pose.y {
            *y = match *y {
                AdjustType::Adjust(y) => AdjustType::Adjust(-y),
                AdjustType::Replace(y) => AdjustType::Replace(-y),
            };
        }
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct StripX<T> {
    pose: T,
}

impl<T> Action for StripX<T> {}

impl StripX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> StripX<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for StripX<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for StripX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for StripX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut pose = self.pose.clone();
        pose.x = None;
        pose
    }
}

impl ActionExec<Stability2Adjust> for StripX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.x = None;
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct ClampX<T> {
    pose: T,
    max: f32,
}

impl<T> Action for ClampX<T> {}

impl ClampX<Stability2Adjust> {
    pub const fn new(max: f32) -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
            max,
        }
    }
}

/*
impl ClampX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(max: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            max,
        }
    }
}
*/

impl<T: Sync + Send + Clone> ActionMod<T> for ClampX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for ClampX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(ref mut x) = self.pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(x.clamp(-self.max, self.max)),
                AdjustType::Replace(x) => AdjustType::Replace(x.clamp(-self.max, self.max)),
            };
        }
        self.pose.clone()
    }
}

impl ActionExec<Stability2Adjust> for ClampX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut pose = self.pose.clone();
        if let Some(ref mut x) = pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(x.clamp(-self.max, self.max)),
                AdjustType::Replace(x) => AdjustType::Replace(x.clamp(-self.max, self.max)),
            };
        }
        pose
    }
}

#[derive(Debug)]
pub struct FlatX<T> {
    pose: T,
}

impl<T> Action for FlatX<T> {}

impl FlatX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl<T: Default> FlatX<T> {
    pub fn new() -> Self {
        Self { pose: T::default() }
    }
}

impl<T: Default> Default for FlatX<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for FlatX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for FlatX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut pose = self.pose.clone();
        if let Some(AdjustType::Replace(val)) = self.pose.x {
            pose.x = if val.is_zero() {
                Some(AdjustType::Replace(0.0))
            } else {
                Some(AdjustType::Replace(-0.3))
            };
        };
        pose
    }
}

impl ActionExec<Stability2Adjust> for FlatX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(AdjustType::Replace(val)) = self.pose.x {
            self.pose.x = if val.is_zero() {
                Some(AdjustType::Replace(0.0))
            } else {
                Some(AdjustType::Replace(-0.3))
            };
        };
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

impl ActionExec<Stability2Adjust> for OffsetToPose<Offset2D<f64>> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut adjust = Stability2Adjust::default();
        adjust.set_x(AdjustType::Replace(*self.offset.x() as f32));
        adjust.set_y(AdjustType::Replace(*self.offset.y() as f32));
        adjust
    }
}

// Messes up type inference for all missions using offset2d
// impl ActionExec<Stability2Adjust> for OffsetToPose<Angle2D<f64>> {
//     async fn execute(&mut self) -> Stability2Adjust {
//         let mut adjust = Stability2Adjust::default();
//         adjust.set_x(AdjustType::Replace(*self.offset.x() as f32));
//         adjust.set_y(AdjustType::Replace(*self.offset.y() as f32));
//         adjust.set_target_yaw(AdjustType::Adjust(*self.offset.angle() as f32));
//         adjust
//     }
// }

#[derive(Debug)]
pub struct BoxToPose<T> {
    input: T,
}

impl<T> Action for BoxToPose<T> {}

impl<T> BoxToPose<T> {
    pub const fn new(input: T) -> Self {
        Self { input }
    }
}

impl<T: Default> Default for BoxToPose<T> {
    fn default() -> Self {
        Self {
            input: T::default(),
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<T> for BoxToPose<T> {
    fn modify(&mut self, input: &T) {
        self.input = input.clone();
    }
}

impl<T: Send + Sync + Clone + Default> ActionMod<Option<T>> for BoxToPose<T> {
    fn modify(&mut self, input: &Option<T>) {
        if let Some(input) = input {
            self.input = input.clone();
        } else {
            self.input = T::default();
        }
    }
}

impl<T: Send + Sync + Clone + Default> ActionMod<anyhow::Result<T>> for BoxToPose<T> {
    fn modify(&mut self, input: &anyhow::Result<T>) {
        if let Ok(input) = input {
            self.input = input.clone();
        } else {
            self.input = T::default();
        }
    }
}

impl ActionExec<Stability2Adjust> for BoxToPose<DrawRect2d> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut adjust = Stability2Adjust::default();
        adjust.set_x(AdjustType::Replace(self.input.x as f32));
        adjust.set_y(AdjustType::Replace(
            ((self.input.width + self.input.height) / 2.0) as f32,
        ));
        adjust
    }
}

impl ActionExec<Stability1Adjust> for BoxToPose<DrawRect2d> {
    async fn execute(&mut self) -> Stability1Adjust {
        let mut adjust = Stability1Adjust::default();
        adjust.set_x(AdjustType::Replace(self.input.x as f32));
        adjust.set_y(AdjustType::Replace(
            ((self.input.width + self.input.height) / 2.0) as f32,
        ));
        adjust
    }
}

/// Modification for a stability assist 1 command
///
/// When values are None, they do not cause adjustments
#[derive(Debug, Clone, Default, Getters)]
pub struct Stability1Adjust {
    x: Option<AdjustType<f32>>,
    y: Option<AdjustType<f32>>,
    target_pitch: Option<AdjustType<f32>>,
    target_roll: Option<AdjustType<f32>>,
    yaw_speed: Option<AdjustType<f32>>,
    target_depth: Option<AdjustType<f32>>,
}

impl Stability1Adjust {
    pub const fn const_default() -> Self {
        Self {
            x: None,
            y: None,
            target_pitch: None,
            target_roll: None,
            yaw_speed: None,
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

    pub fn set_yaw_speed(&mut self, yaw_speed: AdjustType<f32>) -> &Self {
        //self.yaw_speed = Self::bound_speed(Self::address_ieee(yaw_speed));
        self.yaw_speed = Self::address_ieee(yaw_speed);
        self
    }

    pub fn set_target_depth(&mut self, target_depth: AdjustType<f32>) -> &Self {
        self.target_depth = Self::bound_rot(Self::address_ieee(target_depth));
        self
    }
}

/// Stores the command to send to stability assist 2
///
/// If yaw_speed is None, it is set to the current yaw on first execution
#[derive(Debug, Clone)]
pub struct Stability1Pos {
    x: f32,
    y: f32,
    target_pitch: f32,
    target_roll: f32,
    yaw_speed: f32,
    target_depth: f32,
}

impl Stability1Pos {
    pub const fn new(
        x: f32,
        y: f32,
        target_pitch: f32,
        target_roll: f32,
        yaw_speed: f32,
        target_depth: f32,
    ) -> Self {
        Self {
            x,
            y,
            target_pitch,
            target_roll,
            yaw_speed,
            target_depth,
        }
    }

    /// Executes the position in stability assist
    pub async fn exec(&mut self, board: &ControlBoard<WriteHalf<SerialStream>>) -> Result<()> {
        logln!("Stability 1 speed set: {:#?}", self);

        board
            .stability_1_speed_set(
                self.x,
                self.y,
                self.target_pitch,
                self.target_roll,
                self.yaw_speed,
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
    pub fn adjust(&mut self, adjuster: &Stability1Adjust) -> &Self {
        logln!("Stability 2 pre-adjust: {:#?}", self);
        logln!("Adjuster: {:#?}", adjuster);

        self.x = Self::set_speed(self.x, adjuster.x().clone());
        self.y = Self::set_speed(self.y, adjuster.y().clone());

        self.target_pitch = Self::set_rot(self.target_pitch, adjuster.target_pitch().clone());
        self.target_roll = Self::set_rot(self.target_roll, adjuster.target_roll().clone());
        self.target_depth = Self::set_rot(self.target_depth, adjuster.target_depth().clone());

        // Accounting for uninitialized yaw
        //self.yaw_speed = Self::set_speed(self.yaw_speed, adjuster.yaw_speed().clone());
        self.yaw_speed = Self::set_rot(self.yaw_speed, adjuster.yaw_speed().clone());

        logln!("Stability 1 post-adjust: {:#?}", self);
        self
    }

    pub const fn const_default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

impl Default for Stability1Pos {
    fn default() -> Self {
        Self::const_default()
    }
}

#[derive(Debug)]
pub struct Stability1Movement<'a, T> {
    context: &'a T,
    pose: Stability1Pos,
}

impl<T> Action for Stability1Movement<'_, T> {}

impl<'a, T> Stability1Movement<'a, T> {
    pub const fn new(context: &'a T, pose: Stability1Pos) -> Self {
        Self { context, pose }
    }

    pub fn uninitialized(context: &'a T) -> Self {
        Self {
            context,
            pose: Stability1Pos::default(),
        }
    }
}

impl<T> ActionMod<Stability1Pos> for Stability1Movement<'_, T> {
    fn modify(&mut self, input: &Stability1Pos) {
        self.pose = input.clone();
    }
}

impl<T> ActionMod<Stability1Adjust> for Stability1Movement<'_, T> {
    fn modify(&mut self, input: &Stability1Adjust) {
        self.pose.adjust(input);
    }
}

impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for Stability1Movement<'a, T>
{
    async fn execute(&mut self) -> Result<()> {
        self.pose.exec(self.context.get_control_board()).await
    }
}

impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<()> for Stability1Movement<'a, T> {
    async fn execute(&mut self) {
        let _ = self.pose.exec(self.context.get_control_board()).await;
    }
}

impl StripY<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl ActionExec<Stability1Adjust> for StripY<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        self.pose.y = None;
        self.pose.clone()
    }
}

impl StripX<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl ActionExec<Stability1Adjust> for StripX<&Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        let mut pose = self.pose.clone();
        pose.x = None;
        pose
    }
}

impl ActionExec<Stability1Adjust> for StripX<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        self.pose.x = None;
        self.pose.clone()
    }
}

impl FlatX<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new() -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
        }
    }
}

impl ActionExec<Stability1Adjust> for FlatX<&Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        let mut pose = self.pose.clone();
        logln!("Before transform: {:#?}", pose);
        if let Some(AdjustType::Replace(val)) = self.pose.x {
            pose.x = if val.is_zero() {
                Some(AdjustType::Replace(0.0))
            } else {
                Some(AdjustType::Replace(-0.3))
            };
        };
        pose
    }
}

impl ActionExec<Stability1Adjust> for FlatX<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        logln!("Before transform: {:#?}", self.pose);
        if let Some(AdjustType::Replace(val)) = self.pose.x {
            self.pose.x = if val.is_zero() {
                Some(AdjustType::Replace(0.0))
            } else {
                Some(AdjustType::Replace(-0.3))
            };
        };
        self.pose.clone()
    }
}

impl ActionExec<Stability1Adjust> for OffsetToPose<Offset2D<f64>> {
    async fn execute(&mut self) -> Stability1Adjust {
        let mut adjust = Stability1Adjust::default();
        adjust.set_x(AdjustType::Replace(*self.offset.x() as f32));
        adjust.set_y(AdjustType::Replace(*self.offset.y() as f32));
        adjust
    }
}

#[derive(Debug)]
pub struct DefaultGen<T> {
    _phantom: PhantomData<T>,
}

impl<T> Action for DefaultGen<T> {}

impl<T> DefaultGen<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for DefaultGen<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, U: Send + Sync> ActionMod<U> for DefaultGen<T> {
    fn modify(&mut self, _input: &U) {}
}

impl<T: Default + Send + Sync> ActionExec<T> for DefaultGen<T> {
    async fn execute(&mut self) -> T {
        T::default()
    }
}

#[derive(Debug)]
pub struct CautiousConstantX<T> {
    pose: T,
    speed: f32,
}

impl<T> Action for CautiousConstantX<T> {}

impl CautiousConstantX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(speed: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            speed,
        }
    }
}

impl CautiousConstantX<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new(speed: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            speed,
        }
    }
}

impl<T: Default> CautiousConstantX<T> {
    pub fn new(speed: f32) -> Self {
        Self {
            pose: T::default(),
            speed,
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for CautiousConstantX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for CautiousConstantX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(AdjustType::Replace(ref mut x)) = self.pose.x {
            *x = if x.abs() < 0.5 && x.signum() == self.speed.signum() {
                self.speed
            } else {
                0.0
            };
        };
        self.pose.clone()
    }
}

impl ActionExec<Stability1Adjust> for CautiousConstantX<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        if let Some(AdjustType::Replace(ref mut x)) = self.pose.x {
            *x = if !x.is_zero() && x.signum() == self.speed.signum() {
                0.2
            } else {
                0.0
            };
        };
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct MinYaw<T> {
    pose: T,
    speed: f32,
}

impl<T> Action for MinYaw<T> {}

impl MinYaw<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(speed: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            speed,
        }
    }
}

impl MinYaw<&Stability1Adjust> {
    const DEFAULT_POSE: Stability1Adjust = Stability1Adjust::const_default();
    pub const fn new(speed: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            speed,
        }
    }
}

impl<T: Default> MinYaw<T> {
    pub fn new(speed: f32) -> Self {
        Self {
            pose: T::default(),
            speed,
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for MinYaw<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for MinYaw<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(AdjustType::Adjust(ref mut x)) = self.pose.target_yaw {
            if x.is_zero() {
                logln!("ZERO, SETTING MIN SPEED");
                *x = self.speed;
            }
        };
        if self.pose.target_yaw.is_none() {
            logln!("NONE, SETTING MIN SPEED");
            self.pose.target_yaw = Some(AdjustType::Adjust(self.speed));
            self.pose.y = Some(AdjustType::Replace(0.0));
        } else {
            self.pose.y = Some(AdjustType::Replace(0.2));
        }
        self.pose.clone()
    }
}

impl ActionExec<Stability1Adjust> for MinYaw<Stability1Adjust> {
    async fn execute(&mut self) -> Stability1Adjust {
        if let Some(AdjustType::Replace(ref mut x)) = self.pose.yaw_speed {
            if x.is_zero() {
                *x = self.speed;
            }
        };
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct SetX<T> {
    pose: T,
    x: AdjustType<f32>,
}

impl<T> Action for SetX<T> {}

impl SetX<Stability2Adjust> {
    pub const fn new(x: AdjustType<f32>) -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
            x,
        }
    }
}

impl SetX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(x: AdjustType<f32>) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            x,
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for SetX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for SetX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.x = Some(self.x.clone());
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct ConstYaw<T> {
    pose: T,
    yaw: AdjustType<f32>,
}

impl<T> Action for ConstYaw<T> {}

impl ConstYaw<Stability2Adjust> {
    pub const fn new(yaw: AdjustType<f32>) -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
            yaw,
        }
    }
}

impl ConstYaw<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(yaw: AdjustType<f32>) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            yaw,
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for ConstYaw<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for ConstYaw<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        self.pose.target_yaw = Some(self.yaw.clone());
        self.pose.clone()
    }
}

#[derive(Debug)]
pub struct ReplaceX<T> {
    pose: T,
}

impl<T> Action for ReplaceX<T> {}

impl Default for ReplaceX<Stability2Adjust> {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplaceX<Stability2Adjust> {
    pub const fn new() -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
        }
    }
}

/*
impl ReplaceX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(max: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            max,
        }
    }
}
*/

impl<T: Sync + Send + Clone> ActionMod<T> for ReplaceX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for ReplaceX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(ref mut x) = self.pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Replace(x),
                AdjustType::Replace(x) => AdjustType::Replace(x),
            };
        } else {
            self.pose.x = Some(AdjustType::Replace(0.0));
        }
        self.pose.clone()
    }
}

impl ActionExec<Stability2Adjust> for ReplaceX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut pose = self.pose.clone();
        if let Some(ref mut x) = pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Replace(x),
                AdjustType::Replace(x) => AdjustType::Replace(x),
            };
        } else {
            pose.x = Some(AdjustType::Replace(0.0));
        }
        pose
    }
}

#[derive(Debug)]
pub struct MultiplyX<T> {
    pose: T,
    factor: f32,
}

impl<T> Action for MultiplyX<T> {}

impl MultiplyX<Stability2Adjust> {
    pub const fn new(factor: f32) -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
            factor,
        }
    }
}

/*
impl MultiplyX<&Stability2Adjust> {
    const DEFAULT_POSE: Stability2Adjust = Stability2Adjust::const_default();
    pub const fn new(max: f32) -> Self {
        Self {
            pose: &Self::DEFAULT_POSE,
            max,
        }
    }
}
*/

impl<T: Sync + Send + Clone> ActionMod<T> for MultiplyX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for MultiplyX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        if let Some(ref mut x) = self.pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(x * self.factor),
                AdjustType::Replace(x) => AdjustType::Replace(x * self.factor),
            };
        }
        self.pose.clone()
    }
}

impl ActionExec<Stability2Adjust> for MultiplyX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut pose = self.pose.clone();
        if let Some(ref mut x) = pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(x * self.factor),
                AdjustType::Replace(x) => AdjustType::Replace(x * self.factor),
            };
        }
        pose
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub enum Side {
    Red,
    #[default]
    Blue,
}

static SIDE: Mutex<Side> = Mutex::new(Side::Blue);

#[derive(Debug)]
pub struct SetSideRed<T> {
    value: T,
}

impl<T> Action for SetSideRed<T> {}

impl<T: Default> Default for SetSideRed<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default> SetSideRed<T> {
    pub fn new() -> Self {
        Self {
            value: T::default(),
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for SetSideRed<T> {
    fn modify(&mut self, input: &T) {
        self.value = input.clone();
    }
}

impl<T: Send + Sync + Clone> ActionExec<T> for SetSideRed<T> {
    async fn execute(&mut self) -> T {
        logln!("SETTING SIDE TO RED");
        *SIDE.lock().unwrap() = Side::Red;
        self.value.clone()
    }
}

#[derive(Debug)]
pub struct SetSideBlue<T> {
    value: T,
}

impl<T> Action for SetSideBlue<T> {}

impl<T: Default> Default for SetSideBlue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default> SetSideBlue<T> {
    pub fn new() -> Self {
        Self {
            value: T::default(),
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for SetSideBlue<T> {
    fn modify(&mut self, input: &T) {
        self.value = input.clone();
    }
}

impl<T: Send + Sync + Clone> ActionExec<T> for SetSideBlue<T> {
    async fn execute(&mut self) -> T {
        logln!("SETTING SIDE TO BLUE");
        *SIDE.lock().unwrap() = Side::Blue;
        self.value.clone()
    }
}

#[derive(Debug)]
pub struct SideIsRed {}

impl Action for SideIsRed {}

impl Default for SideIsRed {
    fn default() -> Self {
        Self::new()
    }
}

impl SideIsRed {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for SideIsRed {
    fn modify(&mut self, _input: &T) {}
}

impl ActionExec<bool> for SideIsRed {
    async fn execute(&mut self) -> bool {
        *SIDE.lock().unwrap() == Side::Blue
    }
}

#[derive(Debug)]
pub struct SideMult {
    inner: Stability2Adjust,
}

impl Action for SideMult {}

impl Default for SideMult {
    fn default() -> Self {
        Self::new()
    }
}

impl SideMult {
    pub const fn new() -> Self {
        Self {
            inner: Stability2Adjust::const_default(),
        }
    }
}

impl ActionMod<Stability2Adjust> for SideMult {
    fn modify(&mut self, input: &Stability2Adjust) {
        self.inner = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for SideMult {
    async fn execute(&mut self) -> Stability2Adjust {
        let mut inner = self.inner.clone();

        let is_blue = *SIDE.lock().unwrap() == Side::Blue;

        if let Some(ref mut x) = inner.x {
            let x = match x {
                AdjustType::Adjust(x) => x,
                AdjustType::Replace(x) => x,
            };

            if !is_blue {
                *x = -*x;
            }
        };

        if let Some(ref mut yaw) = inner.target_yaw {
            let yaw = match yaw {
                AdjustType::Adjust(yaw) => yaw,
                AdjustType::Replace(yaw) => yaw,
            };
            if !is_blue {
                *yaw = -*yaw;
            }
        };

        inner
    }
}

#[derive(Debug)]
pub struct InvertX<T> {
    pose: T,
}

impl<T> Action for InvertX<T> {}

impl InvertX<Stability2Adjust> {
    pub const fn new() -> Self {
        Self {
            pose: Stability2Adjust::const_default(),
        }
    }
}

impl Default for InvertX<Stability2Adjust> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for InvertX<T> {
    fn modify(&mut self, input: &T) {
        self.pose = input.clone();
    }
}

impl ActionExec<Stability2Adjust> for InvertX<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        logln!("Invert input: {:#?}", self.pose.x);
        if let Some(ref mut x) = self.pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(x.signum() * (1.0 - x.abs())),
                AdjustType::Replace(x) => AdjustType::Replace(x.signum() * (1.0 - x.abs())),
            };
        }
        logln!("Invert output: {:#?}", self.pose.x);
        self.pose.clone()
    }
}

impl ActionExec<Stability2Adjust> for InvertX<&Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        logln!("Invert input: {:#?}", self.pose.x);
        let mut pose = self.pose.clone();
        if let Some(ref mut x) = pose.x {
            *x = match *x {
                AdjustType::Adjust(x) => AdjustType::Adjust(x.signum() * (1.0 - x.abs())),
                AdjustType::Replace(x) => AdjustType::Replace(x.signum() * (1.0 - x.abs())),
            };
        }
        logln!("Invert output: {:#?}", self.pose.x);
        pose
    }
}

#[derive(Debug)]
pub struct GlobalMovement<'a, T> {
    context: &'a T,
    pose: GlobalPos,
}

impl<T> Action for GlobalMovement<'_, T> {}

impl<'a, T> GlobalMovement<'a, T> {
    pub const fn new(context: &'a T, pose: GlobalPos) -> Self {
        Self { context, pose }
    }

    pub fn uninitialized(context: &'a T) -> Self {
        Self {
            context,
            pose: GlobalPos::default(),
        }
    }
}

impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<Result<()>>
    for GlobalMovement<'a, T>
{
    async fn execute(&mut self) -> Result<()> {
        self.pose.exec(self.context.get_control_board()).await
    }
}

impl<'a, T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec<()> for GlobalMovement<'a, T> {
    async fn execute(&mut self) {
        let _ = self.pose.exec(self.context.get_control_board()).await;
    }
}

/// Stores the command to send to stability assist 2
///
/// If target_yaw is None, it is set to the current yaw on first execution
#[derive(Debug, Clone)]
pub struct GlobalPos {
    x: f32,
    y: f32,
    z: f32,
    pitch_speed: f32,
    roll_speed: f32,
    yaw_speed: f32,
}

impl GlobalPos {
    pub const fn new(
        x: f32,
        y: f32,
        z: f32,
        pitch_speed: f32,
        roll_speed: f32,
        yaw_speed: f32,
    ) -> Self {
        Self {
            x,
            y,
            z,
            pitch_speed,
            roll_speed,
            yaw_speed,
        }
    }

    /// Executes the position in stability assist
    pub async fn exec(&mut self, board: &ControlBoard<WriteHalf<SerialStream>>) -> Result<()> {
        board
            .global_speed_set(
                self.x,
                self.y,
                self.z,
                self.pitch_speed,
                self.roll_speed,
                self.yaw_speed,
            )
            .await
    }

    pub const fn const_default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

impl Default for GlobalPos {
    fn default() -> Self {
        Self::const_default()
    }
}

#[derive(Debug)]
pub struct NoAdjust<T> {
    _phantom: PhantomData<T>,
}

impl<T> Action for NoAdjust<T> {}

impl<T> Default for NoAdjust<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> NoAdjust<T> {
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Sync + Send + Clone> ActionMod<T> for NoAdjust<T> {
    fn modify(&mut self, _input: &T) {}
}

impl ActionExec<Stability2Adjust> for NoAdjust<Stability2Adjust> {
    async fn execute(&mut self) -> Stability2Adjust {
        Stability2Adjust::default()
    }
}
