use anyhow::Result;
use opencv::{core::Size, prelude::Mat};

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
    Pole, 
    Blue,
    Gate,
    Middle
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
            0 => Ok(Self::Red),
            1 => Ok(Self::Pole),
            2 => Ok(Self::Blue),
            3 => Ok(Self::Gate),
            4 => Ok(Self::Middle)
            x => Err(TargetError { x }),
        }
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
pub struct GatePoles<T: VisionModel> {
    model: T,
    threshold: f64,
}

impl GatePoles<OnnxModel> {
    pub fn new(model_name: &str, model_size: i32, threshold: f64) -> Result<Self> {
        Ok(Self {
            model: OnnxModel::from_file(model_name, model_size, 5)?,
            threshold,
        })
    }

    pub fn load_640(threshold: f64) -> Self {
        Self {
            model: load_onnx!("models/gate_new_640.onnx", 640, 5),
            threshold,
        }
    }
}

impl Default for GatePoles<OnnxModel> {
    fn default() -> Self {
        Self::load_640(0.7)
    }
}

impl YoloProcessor for GatePoles<OnnxModel> {
    type Target = Target;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Result<Vec<YoloDetection>> {
        self.model.detect_yolo_v5(image, self.threshold)
    }

    fn model_size(&self) -> Size {
        self.model.size()
    }
}
