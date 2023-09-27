use std::{fmt::Debug, hash::Hash};

use super::{
    nn_cv2::{YoloClass, YoloDetection},
    DrawRect2d, VisualDetection, VisualDetector,
};
use anyhow::Result;
use opencv::prelude::Mat;

pub trait YoloTarget: PartialEq + Eq + Hash + Clone + Debug + TryFrom<i32> {}

pub trait YoloProcessor: Debug {
    type Target: PartialEq + Eq + Hash + Clone + Debug + TryFrom<i32>;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Result<Vec<YoloDetection>>;
}

impl<T: YoloProcessor> VisualDetector<f64> for T
where
    <<T as YoloProcessor>::Target as TryFrom<i32>>::Error:
        std::error::Error + Sync + Send + 'static,
{
    type ClassEnum = YoloClass<T::Target>;
    type Position = DrawRect2d;

    fn detect(
        &mut self,
        image: &Mat,
    ) -> Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        self.detect_yolo_v5(image)?
            .into_iter()
            .map(|detection| {
                Ok(VisualDetection {
                    class: YoloClass {
                        identifier: detection.class_id().to_owned().try_into()?,
                        confidence: *detection.confidence(),
                    },
                    position: DrawRect2d {
                        inner: *detection.bounding_box(),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()
    }
}
