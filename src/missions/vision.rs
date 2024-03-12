use std::fmt::Debug;
use std::{iter::Sum, marker::PhantomData, ops::Div};

use super::action::{Action, ActionExec, ActionMod};
use crate::vision::{Draw, Offset2D, RelPos, VisualDetection, VisualDetector};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use num_traits::{Float, FromPrimitive, Num};

use crate::missions::action_context::GetFrontCamMat;
#[cfg(feature = "logging")]
use opencv::{core::Vector, imgcodecs::imwrite};
#[cfg(feature = "logging")]
use std::fs::create_dir_all;
#[cfg(feature = "logging")]
use uuid::Uuid;

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
    > ActionExec for VisionNormOffset<'_, T, U, V>
where
    U::Position: RelPos<Number = V> + Draw,
{
    type Output = Result<Offset2D<V>>;
    async fn execute(&mut self) -> Self::Output {
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
            detections
                .iter()
                .for_each(|x| x.position().draw(&mut mat).unwrap());
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
    > ActionExec for VisionNorm<'_, T, U, V>
where
    U::Position: RelPos<Number = V> + Draw + Send + Sync,
    U::ClassEnum: Send + Sync,
{
    type Output = Result<Vec<VisualDetection<U::ClassEnum, Offset2D<V>>>>;
    async fn execute(&mut self) -> Self::Output {
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
            detections
                .iter()
                .for_each(|x| x.position().draw(&mut mat).unwrap());
            println!("Number of detects: {}", detections.len());
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
    results: anyhow::Result<Vec<VisualDetection<U, V>>>,
    target: T,
}

impl<T, U, V> DetectTarget<T, U, V> {
    pub const fn new(target: T) -> Self {
        Self {
            results: Ok(vec![]),
            target,
        }
    }
}

impl<T, U, V> Action for DetectTarget<T, U, V> {}

#[async_trait]
impl<
        T: Send + Sync + PartialEq,
        U: Send + Sync + Clone + Into<T> + Debug,
        V: Send + Sync + Debug + Clone,
    > ActionExec for DetectTarget<T, U, V>
{
    type Output = anyhow::Result<Vec<VisualDetection<U, V>>>;
    async fn execute(&mut self) -> Self::Output {
        if let Ok(vals) = &self.results {
            let passing_vals: Vec<_> = vals
                .iter()
                .filter(|entry| <U as Into<T>>::into(entry.class().clone()) == self.target)
                .cloned()
                .collect();
            if !passing_vals.is_empty() {
                Ok(passing_vals)
            } else {
                bail!("No valid detections")
            }
        } else {
            bail!("{}", self.results.as_ref().unwrap_err())
        }
    }
}

impl<T, U: Send + Sync + Clone, V: Send + Sync + Clone>
    ActionMod<anyhow::Result<Vec<VisualDetection<U, V>>>> for DetectTarget<T, U, V>
{
    fn modify(&mut self, input: &anyhow::Result<Vec<VisualDetection<U, V>>>) {
        self.results = input
            .as_ref()
            .map(|valid| valid.clone())
            .map_err(|invalid| anyhow!("{}", invalid));
    }
}
