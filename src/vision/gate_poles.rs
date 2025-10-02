use anyhow::Result;
use derive_getters::Getters;
use opencv::core::{multiply, BORDER_CONSTANT, CV_8U};
use opencv::imgproc::{dilate, morphology_default_border_value};
use opencv::{
    core::{merge, split, Point, Size, Vector},
    prelude::{Mat, MatTraitConst},
};

use crate::load_onnx;

use super::{
    nn_cv2::{OnnxModel, VisionModel, YoloClass, YoloDetection},
    yolo_model::YoloProcessor,
};

use core::hash::Hash;
use std::{error::Error, fmt::Display};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Target {
    Red,
    LeftPole,
    RightPole,
    Shark,
    Sawfish,
    Pole,
    Blue,
    Gate,
    Middle,
}

impl From<YoloClass<Target>> for Target {
    fn from(value: YoloClass<Target>) -> Self {
        value.identifier
    }
}

#[derive(Debug)]
pub struct TargetError {
    x: i32,
}

impl Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} is outside known classIDs [0, 3]", self.x)
    }
}

impl Error for TargetError {}

impl TryFrom<i32> for Target {
    type Error = TargetError;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            // 0 => Ok(Self::Red),
            // 1 => Ok(Self::Pole),
            // 2 => Ok(Self::Blue),
            // 3 => Ok(Self::Gate),
            // 4 => Ok(Self::Middle),
            0 => Ok(Self::Gate),
            1 => Ok(Self::Middle),
            2 => Ok(Self::Shark),
            3 => Ok(Self::Sawfish),
            // 4 => Ok(Self::Pole),
            // 5 => Ok(Self::Pole),
            5 => Ok(Self::LeftPole),
            4 => Ok(Self::RightPole),
            x => Err(TargetError { x }),
        }
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Getters)]
pub struct GatePoles<T: VisionModel> {
    model: T,
    threshold: f64,
}

impl GatePoles<OnnxModel> {
    pub fn new(model_name: &str, model_size: i32, threshold: f64) -> Result<Self> {
        let model = OnnxModel::from_file(model_name, model_size, 5)?;

        Ok(Self { model, threshold })
    }

    pub fn load_640(threshold: f64) -> Self {
        let model = load_onnx!("models/new_gate.onnx", 640, 6);

        Self { model, threshold }
    }
}

impl Default for GatePoles<OnnxModel> {
    fn default() -> Self {
        Self::load_640(0.75)
    }
}

impl YoloProcessor for GatePoles<OnnxModel> {
    type Target = Target;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Vec<YoloDetection> {
        let mut channels = Vector::<Mat>::new();
        let _ = split(image, &mut channels).unwrap();
        let b = channels.get(0).unwrap();
        let g = channels.get(1).unwrap();
        let r = channels.get(2).unwrap();
        let mut mult_b = Mat::default();
        let _ = multiply(&b, &1.0, &mut mult_b, 1.0, -1).unwrap();
        let values = Vector::<Mat>::from_iter(vec![b, g, r]);
        let mut output_img = Mat::default();
        let _ = merge(&values, &mut output_img).unwrap();
        let mut dilated = Mat::default();
        // let kernel = Vector::<Vector<i32>>::from_iter(vec![
        //     Vector::<i32>::from_iter(vec![1, 1, 1]),
        //     Vector::<i32>::from_iter(vec![1, 1, 1]),
        //     Vector::<i32>::from_iter(vec![1, 1, 1]),
        // ]);
        // let kernel =
        //     get_structuring_element(MORPH_RECT, Size::new(3, 3), Point::new(-1, -1)).unwrap();
        let kernel = Mat::ones_size(Size::new(3, 3), CV_8U).unwrap();
        let _ = dilate(
            &output_img,
            &mut dilated,
            &kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            morphology_default_border_value().unwrap(),
        )
        .unwrap();

        dbg!(image.dims());
        dbg!(dilated.dims());

        self.model.detect_yolo_v5(&dilated, self.threshold)
    }

    fn model_size(&self) -> Size {
        self.model.size()
    }
}

/*
impl GatePoles<OnnxModel> {
    /// Convert into [`ModelPipelined`].
    ///
    /// See [`ModelPipelined::new`] for arguments.
    pub async fn into_pipelined(
        self,
        model_threads: NonZeroUsize,
        post_processing_threads: NonZeroUsize,
    ) -> ModelPipelined {
        ModelPipelined::new(
            self.model,
            model_threads,
            post_processing_threads,
            self.threshold,
        )
        .await
    }
}
*/
