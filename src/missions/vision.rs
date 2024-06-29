use std::fmt::{Debug, Display};
use std::num::NonZeroUsize;
use std::ops::{Add, Div, Mul};
use std::sync::{Arc, RwLock};
use std::{iter::Sum, marker::PhantomData};

use super::action::{Action, ActionExec, ActionMod};
use super::graph::DotString;
use crate::vision::nn_cv2::{ModelPipelined, VisionModel, YoloClass};
use crate::vision::yolo_model::YoloProcessor;
use crate::vision::{
    Draw, DrawRect2d, MatWrapper, Offset2D, RelPos, VisualDetection, VisualDetector,
};

use anyhow::{anyhow, Result};
use nonzero::nonzero;
use num_traits::{Float, FromPrimitive, Num};
use opencv::core::Mat;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::missions::action_context::GetFrontCamMat;
#[cfg(feature = "logging")]
use opencv::{core::Vector, imgcodecs::imwrite};
#[cfg(feature = "logging")]
use std::fs::create_dir_all;

// Count number of active pipelines, set to true to kill all pipelines.
// All pipelines are cleaned up when count is back to zero.
pub static PIPELINE_KILL: RwLock<(u64, bool)> = RwLock::new((0, false));

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
/// The values are returned without an angle.
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
        println!("Detect attempt: {:#?}", detections);
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

/// Normalizes vision output.
///
/// The relative positions are normalized to [-1, 1] on both axes.
/// The values are returned without an angle.
#[derive(Debug)]
pub struct Norm<U, V> {
    model: U,
    _num: PhantomData<V>,
}

impl<U, V> Norm<U, V> {
    pub const fn new(model: U) -> Self {
        Self {
            model,
            _num: PhantomData,
        }
    }
}

/// Runs a pipelined vision routine to obtain object positions
///
/// The relative positions are normalized to [-1, 1] on both axes.
/// The values are returned without an angle.
#[derive(Debug)]
pub struct VisionPipelinedNorm<T: 'static, U> {
    context: &'static T,
    model: U,
    pipeline: OnceCell<Arc<ModelPipelined>>,
    num_model_threads: NonZeroUsize,
}

impl<T, U> VisionPipelinedNorm<T, U> {
    pub fn new(context: &'static T, model: U, num_model_threads: NonZeroUsize) -> Self {
        Self {
            context,
            model,
            pipeline: OnceCell::new(),
            num_model_threads,
        }
    }
}

impl<T, U> Action for VisionPipelinedNorm<T, U> {}

impl<
        T: GetFrontCamMat + Send + Sync,
        U: VisionModel + YoloProcessor + VisualDetector<f64> + Send + Sync + Clone + 'static,
    > ActionExec<Result<Vec<VisualDetection<YoloClass<U::Target>, DrawRect2d>>>>
    for VisionPipelinedNorm<T, U>
where
    <U as YoloProcessor>::Target: Send + Sync,
    <<U as YoloProcessor>::Target as TryFrom<i32>>::Error: Debug,
    <U as VisionModel>::PostProcessArgs: Send + Sync + Clone,
    <U as VisionModel>::ModelOutput: Send + Sync,
{
    async fn execute(&mut self) -> Result<Vec<VisualDetection<YoloClass<U::Target>, DrawRect2d>>> {
        let model = self.model.clone();
        let context = self.context;
        let num_model_threads = self.num_model_threads;
        let pipeline = self
            .pipeline
            .get_or_init(|| async {
                let pipeline: Arc<ModelPipelined> = Arc::new(
                    ModelPipelined::new(model, num_model_threads, nonzero!(1_usize), 0.7).await,
                );
                let pipeline_clone = pipeline.clone();
                tokio::spawn(async move {
                    PIPELINE_KILL.write().unwrap().0 += 1;
                    while !PIPELINE_KILL.read().unwrap().1 {
                        pipeline_clone.update_mat(MatWrapper(context.get_front_camera_mat().await));
                    }
                    PIPELINE_KILL.write().unwrap().0 -= 1;
                });
                pipeline
            })
            .await;

        #[cfg(feature = "logging")]
        {
            println!("Running detection...");
        }

        #[allow(unused_mut)]
        let mut mat = self.context.get_front_camera_mat().await.clone();
        let detections = pipeline.get_single().await;

        #[cfg(feature = "logging")]
        {
            detections
                .iter()
                .for_each(|x| DrawRect2d::from(*x.bounding_box()).draw(&mut mat).unwrap());
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
                    YoloClass {
                        identifier: (*detect.class_id()).try_into().unwrap(),
                        confidence: *detect.confidence(),
                    },
                    DrawRect2d::from(*detect.bounding_box()),
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
                println!("Passing this: {:#?}", passing_vals);
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

impl<T> Default for Average<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Average<T> {
    pub const fn new() -> Self {
        Self { values: vec![] }
    }
}

impl<T> Action for Average<T> {}

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
        self.values.clone_from(input);
    }
}

impl<T: Send + Sync + Clone> ActionMod<Option<Vec<T>>> for Average<T> {
    fn modify(&mut self, input: &Option<Vec<T>>) {
        if let Some(input) = input {
            self.values.clone_from(input);
        } else {
            self.values = vec![];
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<anyhow::Result<Vec<T>>> for Average<T> {
    fn modify(&mut self, input: &anyhow::Result<Vec<T>>) {
        if let Ok(input) = input {
            self.values.clone_from(input);
        } else {
            self.values = vec![];
        }
    }
}

#[derive(Debug)]
pub struct MidPoint<T> {
    values: Vec<T>,
}

impl<T> Default for MidPoint<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MidPoint<T> {
    pub const fn new() -> Self {
        Self { values: vec![] }
    }
}

impl<T> Action for MidPoint<T> {}

impl ActionExec<Option<Offset2D<f64>>> for MidPoint<Offset2D<f64>> {
    async fn execute(&mut self) -> Option<Offset2D<f64>> {
        if self.values.is_empty() {
            None
        } else {
            let min_x = self
                .values
                .iter()
                .map(|val| val.x())
                .cloned()
                .reduce(f64::min)
                .unwrap();
            let max_x = self
                .values
                .iter()
                .map(|val| val.x())
                .cloned()
                .reduce(f64::max)
                .unwrap();
            let min_y = self
                .values
                .iter()
                .map(|val| val.y())
                .cloned()
                .reduce(f64::min)
                .unwrap();
            let max_y = self
                .values
                .iter()
                .map(|val| val.y())
                .cloned()
                .reduce(f64::max)
                .unwrap();

            let val = Some(Offset2D::new((max_x + min_x) / 2.0, (max_y + min_y) / 2.0));
            println!("Processed this: {:#?}", val);
            val
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<Vec<T>> for MidPoint<T> {
    fn modify(&mut self, input: &Vec<T>) {
        self.values.clone_from(input);
    }
}

impl<T: Send + Sync + Clone> ActionMod<Option<Vec<T>>> for MidPoint<T> {
    fn modify(&mut self, input: &Option<Vec<T>>) {
        if let Some(input) = input {
            self.values.clone_from(input);
        } else {
            self.values = vec![];
        }
    }
}

impl<T: Send + Sync + Clone> ActionMod<anyhow::Result<Vec<T>>> for MidPoint<T> {
    fn modify(&mut self, input: &anyhow::Result<Vec<T>>) {
        if let Ok(input) = input {
            self.values.clone_from(input);
        } else {
            self.values = vec![];
        }
    }
}

#[derive(Debug)]
pub struct ExtractPosition<T, U> {
    values: Vec<VisualDetection<T, U>>,
}

impl<T, U> Default for ExtractPosition<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, U> ExtractPosition<T, U> {
    pub const fn new() -> Self {
        Self { values: vec![] }
    }
}

impl<T, U> Action for ExtractPosition<T, U> {}

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
        self.values.clone_from(input);
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone> ActionMod<VisualDetection<T, U>>
    for ExtractPosition<T, U>
{
    fn modify(&mut self, input: &VisualDetection<T, U>) {
        self.values = vec![input.clone()];
    }
}

#[derive(Debug)]
pub struct ToOffset<T, U> {
    values: Vec<VisualDetection<T, U>>,
}

impl<T, U> Default for ToOffset<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, U> ToOffset<T, U> {
    pub const fn new() -> Self {
        Self { values: vec![] }
    }
}

impl<T, U> Action for ToOffset<T, U> {}

impl<T: Send + Sync, U: Send + Sync + RelPos> ActionExec<Vec<Offset2D<U::Number>>>
    for ToOffset<T, U>
where
    <U as RelPos>::Number: Send + Sync,
{
    async fn execute(&mut self) -> Vec<Offset2D<U::Number>> {
        self.values
            .iter()
            .map(|val| val.position().offset())
            .collect()
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone> ActionMod<Vec<VisualDetection<T, U>>>
    for ToOffset<T, U>
{
    fn modify(&mut self, input: &Vec<VisualDetection<T, U>>) {
        self.values.clone_from(input);
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone> ActionMod<VisualDetection<T, U>>
    for ToOffset<T, U>
{
    fn modify(&mut self, input: &VisualDetection<T, U>) {
        self.values = vec![input.clone()];
    }
}

#[derive(Debug)]
pub struct OffsetClass<T, U, V> {
    values: Vec<VisualDetection<T, U>>,
    class: V,
    offset: U,
}

impl<T, U, V> OffsetClass<T, U, V> {
    pub const fn new(class: V, offset: U) -> Self {
        Self {
            values: vec![],
            class,
            offset,
        }
    }
}

impl<T, U, V> Action for OffsetClass<T, U, V> {}

impl<
        T: Send + Sync + Clone + Into<V>,
        U: Send + Sync + Clone + Add<Output = U>,
        V: Send + Sync + PartialEq,
    > ActionExec<Vec<VisualDetection<T, U>>> for OffsetClass<T, U, V>
where
    T: PartialEq<T>,
{
    async fn execute(&mut self) -> Vec<VisualDetection<T, U>> {
        self.values
            .iter()
            .map(|x| {
                let offset = if x.class().clone().into() == self.class {
                    x.position().clone() + self.offset.clone()
                } else {
                    x.position().clone()
                };
                VisualDetection::new(x.class().clone(), offset)
            })
            .collect()
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone, V> ActionMod<Vec<VisualDetection<T, U>>>
    for OffsetClass<T, U, V>
{
    fn modify(&mut self, input: &Vec<VisualDetection<T, U>>) {
        self.values.clone_from(input);
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone, V> ActionMod<VisualDetection<T, U>>
    for OffsetClass<T, U, V>
{
    fn modify(&mut self, input: &VisualDetection<T, U>) {
        self.values = vec![input.clone()];
    }
}
