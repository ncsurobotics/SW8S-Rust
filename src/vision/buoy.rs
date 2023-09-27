use anyhow::Result;
use opencv::prelude::Mat;

use crate::load_onnx;

use super::{
    nn_cv2::{OnnxModel, VisionModel, YoloDetection},
    yolo_model::{YoloProcessor, YoloTarget},
};

use core::hash::Hash;
use std::{error::Error, fmt::Display};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Target {
    Earth1,
    Earth2,
    Abydos1,
    Abydos2,
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

impl YoloTarget for Target {}

impl TryFrom<i32> for Target {
    type Error = TargetError;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Earth1),
            1 => Ok(Self::Earth2),
            2 => Ok(Self::Abydos1),
            3 => Ok(Self::Abydos2),
            x => Err(TargetError { x }),
        }
    }
}

#[derive(Debug)]
pub struct Buoy<T: VisionModel> {
    model: T,
    threshold: f64,
}

impl<T: VisionModel> Buoy<T> {}

impl Buoy<OnnxModel> {
    pub fn new(model_name: &str, model_size: i32, threshold: f64) -> Result<Self> {
        Ok(Self {
            model: OnnxModel::from_file(model_name, model_size, 4)?,
            threshold,
        })
    }

    pub fn load_320(threshold: f64) -> Self {
        Self {
            model: load_onnx!("models/buoy_320.onnx", 320, 4),
            threshold,
        }
    }

    pub fn load_640(threshold: f64) -> Self {
        Self {
            model: load_onnx!("models/buoy_640.onnx", 640, 4),
            threshold,
        }
    }
}

impl Default for Buoy<OnnxModel> {
    fn default() -> Self {
        Self::load_320(0.7)
    }
}

impl YoloProcessor for Buoy<OnnxModel> {
    type Target = Target;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Result<Vec<YoloDetection>> {
        self.model.detect_yolo_v5(image, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use opencv::{
        core::Vector,
        imgcodecs::{imread, imwrite, IMREAD_COLOR},
    };

    use crate::vision::{Draw, VisualDetector};
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn detect_single() {
        let mut image = imread("tests/vision/resources/buoy_images/1.jpeg", IMREAD_COLOR).unwrap();
        let detections = Buoy::default().detect(&image).unwrap();
        let detect_unique: Vec<_> = detections.iter().unique().collect();

        detect_unique
            .iter()
            .for_each(|result| result.position().draw(&mut image).unwrap());

        println!("Detections: {:#?}", detect_unique);
        imwrite(
            "tests/vision/output/buoy_images/1.jpeg",
            &image,
            &Vector::default(),
        )
        .unwrap();

        let abydos_1_pos = detect_unique
            .iter()
            .find(|&result| *result.class() == Target::Abydos1)
            .unwrap()
            .position();
        assert_approx_eq!(abydos_1_pos.x, 134.9113845825195);
        assert_approx_eq!(abydos_1_pos.y, 163.99715423583984);
        assert_approx_eq!(abydos_1_pos.width, 149.86732482910156);
        assert_approx_eq!(abydos_1_pos.height, 141.14679336547852);
    }
}
