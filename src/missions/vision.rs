use std::fmt::{Debug, Display};
use std::ops::{Div, Mul};
use std::{iter::Sum, marker::PhantomData};

use super::action::{Action, ActionExec, ActionMod};
use super::graph::DotString;
use crate::vision::{Draw, Offset2D, RelPos, VisualDetection, VisualDetector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use num_traits::{Float, FromPrimitive, Num};
use opencv::core::Mat;
use uuid::Uuid;

use crate::missions::action_context::GetFrontCamMat;
#[cfg(feature = "logging")]
use opencv::{core::Vector, imgcodecs::imwrite};
#[cfg(feature = "logging")]
use std::fs::create_dir_all;

/// Runs a vision routine to obtain the average of object positions
///
/// The relative position is normalized to [-1, 1] on both axes
#[derive(Debug)]
pub struct VisionNormOffset<'a, T, U, V> {
    context: &'a T,
    model: U,
    _num: PhantomData<V>,
}

impl<'a, T, U, V> VisionNormOffset<'a, T, U, V> {
    pub const fn new(context: &'a T, model: U) -> Self {
        Self {
            context,
            model,
            _num: PhantomData,
        }
    }
}

impl<T, U, V> Action for VisionNormOffset<'_, T, U, V> {}

#[async_trait]
impl<
        T: GetFrontCamMat + Send + Sync,
        V: Num + Float + FromPrimitive + Send + Sync,
        U: VisualDetector<V> + Send + Sync,
    > ActionExec<Result<Offset2D<V>>> for VisionNormOffset<'_, T, U, V>
where
    U::Position: RelPos<Number = V> + for<'a> Mul<&'a Mat, Output = U::Position>,
    VisualDetection<U::ClassEnum, U::Position>: Draw,
{
    async fn execute(&mut self) -> Result<Offset2D<V>> {
        #[cfg(feature = "logging")]
        {
            println!("Running detection...");
        }

        #[allow(unused_mut)]
        let mut mat = self.context.get_front_camera_mat().await.clone();
        let detections = self.model.detect(&mat);
        #[cfg(feature = "logging")]
        println!("Detect attempt: {}", detections.is_ok());
        let detections = detections?;
        #[cfg(feature = "logging")]
        {
            detections.iter().for_each(|x| {
                let x = VisualDetection::new(
                    x.class().clone(),
                    self.model.normalize(x.position()) * &mat,
                );
                x.draw(&mut mat).unwrap()
            });
            println!("Number of detects: {}", detections.len());
            let _ = create_dir_all("/tmp/detect");
            imwrite(
                &("/tmp/detect/".to_string() + &Uuid::new_v4().to_string() + ".jpeg"),
                &mat,
                &Vector::default(),
            )
            .unwrap();
        }

        let positions: Vec<_> = detections
            .iter()
            .map(|detect| self.model.normalize(detect.position()))
            .map(|detect| detect.offset())
            .collect();

        let positions_len = positions.len();

        let offset = positions.into_iter().sum::<Offset2D<V>>() / positions_len;
        if offset.x().is_nan() || offset.y().is_nan() {
            Err(anyhow!("NaN values"))
        } else {
            Ok(offset)
        }
    }
}

/// Runs a vision routine to obtain object positions
///
/// The relative positions are normalized to [-1, 1] on both axes.
/// The values are returned without an angle
#[derive(Debug)]
pub struct VisionNorm<'a, T, U, V> {
    context: &'a T,
    model: U,
    _num: PhantomData<V>,
}

impl<'a, T, U, V> VisionNorm<'a, T, U, V> {
    pub const fn new(context: &'a T, model: U) -> Self {
        Self {
            context,
            model,
            _num: PhantomData,
        }
    }
}

impl<T, U, V> Action for VisionNorm<'_, T, U, V> {}

#[async_trait]
impl<
        T: GetFrontCamMat + Send + Sync,
        V: Num + Float + FromPrimitive + Send + Sync,
        U: VisualDetector<V> + Send + Sync,
    > ActionExec<Result<Vec<VisualDetection<U::ClassEnum, Offset2D<V>>>>>
    for VisionNorm<'_, T, U, V>
where
    U::Position: RelPos<Number = V> + Debug + for<'a> Mul<&'a Mat, Output = U::Position>,
    VisualDetection<U::ClassEnum, U::Position>: Draw,
    U::ClassEnum: Send + Sync + Debug,
{
    async fn execute(&mut self) -> Result<Vec<VisualDetection<U::ClassEnum, Offset2D<V>>>> {
        #[cfg(feature = "logging")]
        {
            println!("Running detection...");
        }

        #[allow(unused_mut)]
        let mut mat = self.context.get_front_camera_mat().await.clone();
        let detections = self.model.detect(&mat);
        #[cfg(feature = "logging")]
        println!("Detect attempt");
        //println!("Detect attempt: {:#?}", detections);
        let detections = detections?;
        #[cfg(feature = "logging")]
        {
            detections.iter().for_each(|x| {
                let x = VisualDetection::new(
                    x.class().clone(),
                    self.model.normalize(x.position()) * &mat,
                );
                x.draw(&mut mat).unwrap()
            });
            create_dir_all("/tmp/detect").unwrap();
            imwrite(
                &("/tmp/detect/".to_string() + &Uuid::new_v4().to_string() + ".jpeg"),
                &mat,
                &Vector::default(),
            )
            .unwrap();
        }

        Ok(detections
            .into_iter()
            .map(|detect| {
                VisualDetection::new(
                    detect.class().clone(),
                    self.model.normalize(detect.position()).offset(),
                )
            })
            .collect())
    }
}

#[derive(Debug)]
pub struct DetectTarget<T, U, V> {
    results: Option<Vec<VisualDetection<U, V>>>,
    target: T,
}

impl<T, U, V> DetectTarget<T, U, V> {
    pub const fn new(target: T) -> Self {
        Self {
            results: None,
            target,
        }
    }
}

impl<T: Display, U, V> Action for DetectTarget<T, U, V> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let id = Uuid::new_v4();
        DotString {
            head_ids: vec![id],
            tail_ids: vec![id],
            body: format!(
                "\"{}\" [label = \"Detect {}\", margin = 0];\n",
                id, self.target
            ),
        }
    }
}

#[async_trait]
impl<
        T: Send + Sync + PartialEq + Display,
        U: Send + Sync + Clone + Into<T> + Debug,
        V: Send + Sync + Debug + Clone,
    > ActionExec<Option<Vec<VisualDetection<U, V>>>> for DetectTarget<T, U, V>
{
    async fn execute(&mut self) -> Option<Vec<VisualDetection<U, V>>> {
        if let Some(vals) = &self.results {
            let passing_vals: Vec<_> = vals
                .iter()
                .filter(|entry| <U as Into<T>>::into(entry.class().clone()) == self.target)
                .cloned()
                .collect();
            if !passing_vals.is_empty() {
                Some(passing_vals)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<T: Display, U: Send + Sync + Clone, V: Send + Sync + Clone>
    ActionMod<anyhow::Result<Vec<VisualDetection<U, V>>>> for DetectTarget<T, U, V>
{
    fn modify(&mut self, input: &anyhow::Result<Vec<VisualDetection<U, V>>>) {
        #[allow(clippy::all)]
        {
            self.results = input.as_ref().map(|valid| valid.clone()).ok()
        }
    }
}

impl<T: Display, U: Send + Sync + Clone, V: Send + Sync + Clone>
    ActionMod<Option<Vec<VisualDetection<U, V>>>> for DetectTarget<T, U, V>
{
    fn modify(&mut self, input: &Option<Vec<VisualDetection<U, V>>>) {
        self.results = input.as_ref().cloned();
    }
}

#[derive(Debug)]
pub struct Average<T> {
    values: Vec<T>,
}

impl<T> Average<T> {
    pub const fn new() -> Self {
        Self { values: vec![] }
    }
}

impl<T> Action for Average<T> {}

#[async_trait]
impl<T: Send + Sync + Clone + Sum + Div<usize, Output = T>> ActionExec<Option<T>> for Average<T> {
    async fn execute(&mut self) -> Option<T> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.values.clone().into_iter().sum::<T>() / self.values.len())
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<Vec<T>> for Average<T> {
    fn modify(&mut self, input: &Vec<T>) {
        self.values = input.clone();
    }
}

impl<T: Send + Sync + Clone> ActionMod<Option<Vec<T>>> for Average<T> {
    fn modify(&mut self, input: &Option<Vec<T>>) {
        if let Some(input) = input {
            self.values = input.clone();
        } else {
            self.values = vec![];
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<anyhow::Result<Vec<T>>> for Average<T> {
    fn modify(&mut self, input: &anyhow::Result<Vec<T>>) {
        if let Ok(input) = input {
            self.values = input.clone();
        } else {
            self.values = vec![];
        }
    }
}

#[derive(Debug)]
pub struct ExtractPosition<T, U> {
    values: Vec<VisualDetection<T, U>>,
}

impl<T, U> ExtractPosition<T, U> {
    pub const fn new() -> Self {
        Self { values: vec![] }
    }
}

impl<T, U> Action for ExtractPosition<T, U> {}

#[async_trait]
impl<T: Send + Sync, U: Send + Sync + Clone> ActionExec<Vec<U>> for ExtractPosition<T, U> {
    async fn execute(&mut self) -> Vec<U> {
        self.values
            .iter()
            .map(|val| val.position().clone())
            .collect()
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone> ActionMod<Vec<VisualDetection<T, U>>>
    for ExtractPosition<T, U>
{
    fn modify(&mut self, input: &Vec<VisualDetection<T, U>>) {
        self.values = input.clone();
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone> ActionMod<VisualDetection<T, U>>
    for ExtractPosition<T, U>
{
    fn modify(&mut self, input: &VisualDetection<T, U>) {
        self.values = vec![input.clone()];
    }
}
