use anyhow::{anyhow, Result};
use derive_getters::Getters;
use itertools::Itertools;
use num_traits::{zero, FromPrimitive, Num};
use opencv::{
    core::{MatTraitConst, Rect2d, Scalar},
    imgproc::{self, LINE_8},
    prelude::Mat,
};
use std::{
    fmt::Debug,
    hash::Hash,
    iter::Sum,
    ops::{Add, Deref, Div, Mul},
};

pub mod buoy;
pub mod buoy_model;
pub mod gate;
pub mod gate_poles;
pub mod image_prep;
pub mod nn_cv2;
pub mod path;
pub mod pca;
pub mod yolo_model;

pub trait Draw {
    /// Draws self on top of `canvas`
    ///
    /// # Arguments
    /// `canvas` - base image
    fn draw(&self, canvas: &mut Mat) -> Result<()>;

    fn draw_all<I, T>(draw_it: I, canvas: &mut Mat) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Draw,
    {
        draw_it.into_iter().try_for_each(|item| item.draw(canvas))
    }
}

/// Holds x and y offset of object in frame
#[derive(Debug, Getters, Clone, Copy, Default)]
pub struct Offset2D<T: Num> {
    x: T,
    y: T,
}

pub trait RelPos {
    type Number: Num;
    fn offset(&self) -> Offset2D<Self::Number>;
}

impl<T: Num> Add for Offset2D<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Num> Sum for Offset2D<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, cur| acc + cur).unwrap_or(Self {
            x: zero(),
            y: zero(),
        })
    }
}

impl<T: Num + FromPrimitive> Div<usize> for Offset2D<T> {
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        Self {
            x: self.x / T::from_usize(rhs).unwrap(),
            y: self.y / T::from_usize(rhs).unwrap(),
        }
    }
}

/// Holds x, y, and angle offset of object in frame
#[derive(Debug, Getters)]
pub struct Angle2D<T: Num> {
    x: T,
    y: T,
    angle: T,
}

impl<T: Num> Add for Angle2D<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            angle: self.angle + rhs.angle,
        }
    }
}

impl<T: Num + From<usize>> Div<usize> for Angle2D<T> {
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        Self {
            x: self.x / rhs.into(),
            y: self.y / rhs.into(),
            angle: self.angle / rhs.into(),
        }
    }
}

impl<T: Num> From<Angle2D<T>> for Offset2D<T> {
    fn from(val: Angle2D<T>) -> Self {
        Offset2D { x: val.x, y: val.y }
    }
}

impl<T: Num> From<Offset2D<T>> for Angle2D<T> {
    fn from(val: Offset2D<T>) -> Self {
        Angle2D {
            x: val.x,
            y: val.y,
            angle: T::zero(),
        }
    }
}

pub trait RelPosAngle {
    type Number: Num;
    fn offset_angle(&self) -> Angle2D<Self::Number>;
}

impl<T: RelPosAngle> RelPos for T {
    type Number = T::Number;
    fn offset(&self) -> Offset2D<Self::Number> {
        self.offset_angle().into()
    }
}

pub fn mean<'a, T>(values: &'a [T]) -> T
where
    &'a T: Div<usize, Output = T> + Add<Output = &'a T>,
{
    values
        .iter()
        .reduce(|x, y| x + y)
        .expect("Cannot calculate mean of 0 numbers")
        / values.len()
}

pub trait VisualDetector<T: Num>: Debug {
    type ClassEnum: PartialEq + Eq + Hash + Clone;
    type Position: RelPos<Number = f64> + Clone;

    fn detect(
        &mut self,
        image: &Mat,
    ) -> Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>>;

    fn detect_unique(
        &mut self,
        image: &Mat,
    ) -> Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        Ok(self.detect(image)?.into_iter().unique().collect())
    }

    fn detect_class(
        &mut self,
        image: &Mat,
        target: Self::ClassEnum,
    ) -> Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        Ok(self
            .detect(image)?
            .into_iter()
            .filter(|result| result.class == target)
            .collect())
    }

    /// Adjusts position to [-1, 1] on both axes
    fn normalize(&mut self, pos: &Self::Position) -> Self::Position;
}

#[derive(Debug, Clone, Getters)]
pub struct VisualDetection<T, U> {
    class: T,
    position: U,
}

impl<T, U> VisualDetection<T, U> {
    pub fn new(class: T, position: U) -> Self {
        Self { class, position }
    }
}

impl<T: PartialEq, U> PartialEq<T> for VisualDetection<T, U> {
    fn eq(&self, other: &T) -> bool {
        self.class == *other
    }
}

impl<T: PartialEq, U> PartialEq<Self> for VisualDetection<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class
    }
}

impl<T: PartialEq, U> Eq for VisualDetection<T, U> {}

impl<T: Hash, U> Hash for VisualDetection<T, U> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.class.hash(state)
    }
}

impl RelPos for Offset2D<f64> {
    type Number = f64;

    fn offset(&self) -> Offset2D<Self::Number> {
        Offset2D {
            x: self.x,
            y: self.y,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DrawRect2d {
    inner: Rect2d,
}

impl Deref for DrawRect2d {
    type Target = Rect2d;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl RelPos for DrawRect2d {
    type Number = f64;

    fn offset(&self) -> Offset2D<Self::Number> {
        Offset2D {
            x: self.inner.x + (self.inner.width / 2.0),
            y: self.inner.y + (self.inner.height / 2.0),
        }
    }
}

impl Draw for DrawRect2d {
    fn draw(&self, canvas: &mut Mat) -> Result<()> {
        imgproc::rectangle(
            canvas,
            self.inner
                .to()
                .ok_or(anyhow!("f64 outside bounds of i32"))?,
            Scalar::from((0.0, 0.0, 255.0)),
            2,
            LINE_8,
            0,
        )?;
        Ok(())
    }
}

impl Mul<&Mat> for DrawRect2d {
    type Output = Self;

    fn mul(self, rhs: &Mat) -> Self::Output {
        let size = rhs.size().unwrap();
        Self {
            inner: Rect2d {
                width: (self.width + 0.5) * (size.width as f64),
                height: (self.height + 0.5) * (size.height as f64),
                x: (self.x + 1.0) * (size.width as f64),
                y: (self.y + 1.0) * (size.height as f64),
            },
        }
    }
}
