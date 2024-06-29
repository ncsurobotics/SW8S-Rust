use anyhow::Result;
use derive_getters::Getters;
use opencv::{core::Size, prelude::Mat};

use crate::load_onnx;

use super::{
    nn_cv2::{ModelPipelined, OnnxModel, VisionModel, YoloClass, YoloDetection},
    yolo_model::YoloProcessor,
};

use core::hash::Hash;
use std::{error::Error, fmt::Display, num::NonZeroUsize};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Target {
    Buoy,
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
        write!(f, "{} is outside known classIDs [0, 0]", self.x)
    }
}

impl Error for TargetError {}

impl TryFrom<i32> for Target {
    type Error = TargetError;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Buoy),
            x => Err(TargetError { x }),
        }
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Getters)]
pub struct BuoyModel<T: VisionModel> {
    model: T,
    threshold: f64,
}

impl BuoyModel<OnnxModel> {
    pub fn new(model_name: &str, model_size: i32, threshold: f64) -> Result<Self> {
        Ok(Self {
            model: OnnxModel::from_file(model_name, model_size, 1)?,
            threshold,
        })
    }

    pub fn load_640(threshold: f64) -> Self {
        Self {
            model: load_onnx!("models/buoy_single_class_640.onnx", 640, 1),
            threshold,
        }
    }
}

impl Default for BuoyModel<OnnxModel> {
    fn default() -> Self {
        Self::load_640(0.7)
    }
}

impl YoloProcessor for BuoyModel<OnnxModel> {
    type Target = Target;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Vec<YoloDetection> {
        self.model.detect_yolo_v5(image, self.threshold)
    }

    fn model_size(&self) -> Size {
        self.model.size()
    }
}

impl BuoyModel<OnnxModel> {
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

impl VisionModel for BuoyModel<OnnxModel> {
    type ModelOutput = <OnnxModel as VisionModel>::ModelOutput;
    type PostProcessArgs = <OnnxModel as VisionModel>::PostProcessArgs;

    fn detect_yolo_v5(&mut self, image: &Mat, threshold: f64) -> Vec<YoloDetection> {
        self.model.detect_yolo_v5(image, threshold)
    }
    fn forward(&mut self, image: &Mat) -> Self::ModelOutput {
        self.model.forward(image)
    }
    fn post_process_args(&self) -> Self::PostProcessArgs {
        self.model.post_process_args()
    }
    fn post_process(
        args: Self::PostProcessArgs,
        output: Self::ModelOutput,
        threshold: f64,
    ) -> Vec<YoloDetection> {
        OnnxModel::post_process(args, output, threshold)
    }
    fn size(&self) -> Size {
        self.model.size()
    }
}
