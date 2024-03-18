#[cfg(feature = "logging")]
use std::fs::create_dir_all;
use std::marker::PhantomData;

use anyhow::Result;
use async_trait::async_trait;
use num_traits::{FromPrimitive, Num};
#[cfg(feature = "logging")]
use opencv::{core::Vector, imgcodecs::imwrite};
#[cfg(feature = "logging")]
use uuid::Uuid;

use crate::missions::action_context::GetFrontCamMat;
use crate::vision::{Draw, Offset2D, RelPos, VisualDetection, VisualDetector};
use crate::vision::gate_poles::GatePoles;
use crate::vision::nn_cv2::OnnxModel;

use super::action::{Action, ActionExec, ActionMod};

/// Runs a vision routine to obtain object position
///
/// The relative position is normalized to [-1, 1] on both axes
#[derive(Debug)]
pub struct VisionNormOffset<U, V> {
    model: GatePoles<OnnxModel>,
    detections: Vec<VisualDetection<<U as VisualDetector<V>>::ClassEnum, <U as VisualDetector<V>>::Position>>,
}

impl<U, V> VisionNormOffset<U, V> {
    pub const fn new() -> Self {
        Self {
            model: Default::default(),
            detections,
        }
    }
}

impl<U, V> Action for VisionNormOffset<U, V> {}

#[async_trait]
impl<U, V> ActionExec for VisionNormOffset<U, V> {
    type Output = Result<Offset2D<V>>;

    async fn execute(&mut self) -> Self::Output {
        let positions: Vec<_> = self.detections
            .iter()
            .map(|detect| detect.offset())
            .collect();

        let positions_len = positions.len();

        Ok(positions.into_iter().sum::<Offset2D<V>>() / positions_len)
    }
}

impl<U, V> ActionMod<
    Vec<VisualDetection<<U as VisualDetector<V>>::ClassEnum, <U as VisualDetector<V>>::Position>>
> for VisionNormOffset<U, V>
    where
        <U as VisualDetector<V>>::ClassEnum: Sync + Send,
        <U as VisualDetector<V>>::Position: Sync + Send
{
    fn modify(&mut self, input: &Vec<VisualDetection<<U as VisualDetector<V>>::ClassEnum, <U as VisualDetector<V>>::Position>>) {
        self.detections = input.clone();
    }
}

#[derive(Debug)]
pub struct VisionDetect<'a, T, U, V> {
    context: &'a T,
    model: U,
    _num: PhantomData<V>,
}

impl<'a, T, U, V> VisionDetect<'a, T, U, V> {
    pub const fn new(context: &'a T, model: U) -> Self {
        Self {
            context,
            model,
            _num: PhantomData,
        }
    }
}

impl<T, U, V> Action for VisionDetect<'_, T, U, V> {}

#[async_trait]
impl<
    T: GetFrontCamMat + Send + Sync,
    V: Num + FromPrimitive + Send + Sync,
    U: VisualDetector<V> + Send + Sync,
> ActionExec for VisionDetect<'_, T, U, V>
    where
        U::Position: RelPos<Number=V> + Draw,
        <U as VisualDetector<V>>::ClassEnum: Sync + Send,
        <U as VisualDetector<V>>::Position: Sync + Send
{
    type Output = Result<Vec<VisualDetection<<U as VisualDetector<V>>::ClassEnum, <U as VisualDetector<V>>::Position>>>;
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

        let norm_detections: Vec<VisualDetection<<U as VisualDetector<V>>::ClassEnum, <U as VisualDetector<V>>::Position>> = detections
            .iter()
            .map(|detect| self.model.normalize(detect.position()))
            .collect();

        Ok(norm_detections)
    }
}


#[derive(Debug)]
pub struct PoleDetect<'a, T, U, V> {
    context: &'a T,
    model: U,
    _num: PhantomData<V>,
}

impl<'a, T, U, V> PoleDetect<'a, T, U, V> {
    pub const fn new(context: &'a T, model: U) -> Self {
        Self {
            context,
            model,
            _num: PhantomData,
        }
    }
}

impl<T, U, V> Action for PoleDetect<'_, T, U, V> {}

#[async_trait]
impl<T: GetFrontCamMat + Send + Sync, V: Num + FromPrimitive + Send + Sync, U: VisualDetector<V> + Send + Sync> ActionExec
for PoleDetect<'_, T, U, V> {
    type Output = Result<bool>;

    async fn execute(&mut self) -> Self::Output {
        todo!()
    }
}
