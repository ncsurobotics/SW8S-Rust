use std::{fmt::Debug, hash::Hash};

use super::{nn_cv2::YoloDetection, DrawRect2d, VisualDetection, VisualDetector};
use anyhow::Result;
use opencv::prelude::Mat;

pub trait YoloProcessor: Debug {
    type Target: PartialEq + Eq + Hash + Clone + TryFrom<i32, Error = anyhow::Error> + Debug;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Result<Vec<YoloDetection>>;
}

impl<T: YoloProcessor> VisualDetector<f64> for T {
    type ClassEnum = T::Target;
    type Position = DrawRect2d;

    fn detect(
        &mut self,
        image: &Mat,
    ) -> Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        self.detect_yolo_v5(image)?
            .iter()
            .map(|detection| {
                Ok(VisualDetection {
                    class: T::Target::try_from(*detection.class_id())?,
                    confidence: *detection.confidence(),
                    position: DrawRect2d {
                        inner: *detection.bounding_box(),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()
    }
}
