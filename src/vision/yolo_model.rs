use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

use super::{
    nn_cv2::{YoloClass, YoloDetection},
    Draw, DrawRect2d, RelPos, VisualDetection, VisualDetector,
};
use anyhow::Result;
use opencv::{
    core::{Point, Rect2d, Scalar, Size},
    imgproc::{self, LINE_AA},
    prelude::Mat,
};

pub trait YoloTarget: PartialEq + Eq + Hash + Clone + Debug + TryFrom<i32> {}

pub trait YoloProcessor: Debug {
    type Target: PartialEq + Eq + Hash + Clone + Debug + TryFrom<i32>;

    fn detect_yolo_v5(&mut self, image: &Mat) -> Vec<YoloDetection>;
    fn model_size(&self) -> Size;
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
        Ok(self
            .detect_yolo_v5(image)
            .into_iter()
            .map(|detection| VisualDetection {
                class: YoloClass {
                    identifier: detection.class_id().to_owned().try_into().unwrap(),
                    confidence: *detection.confidence(),
                },
                position: DrawRect2d {
                    inner: *detection.bounding_box(),
                },
            })
            .collect::<Vec<_>>())
    }

    fn normalize(&mut self, pos: &Self::Position) -> Self::Position {
        // Temporary constants
        const IMAGE_WIDTH: f64 = 640.0;
        const IMAGE_HEIGHT: f64 = 480.0;
        Self::Position {
            inner: Rect2d::new(
                ((pos.inner.x / IMAGE_WIDTH) - 0.5) * 2.0,
                ((pos.inner.y / IMAGE_HEIGHT) - 0.5) * 2.0,
                pos.inner.width / IMAGE_WIDTH,
                pos.inner.height / IMAGE_HEIGHT,
            ),
        }
    }
}

impl<T: Display> Draw for VisualDetection<YoloClass<T>, DrawRect2d> {
    fn draw(&self, canvas: &mut Mat) -> Result<()> {
        self.position.draw(canvas)?;

        let center_point = self.position.offset();
        imgproc::put_text(
            canvas,
            &self.class.identifier.to_string(),
            Point::new(
                // Adjust x to 1/4 from left b/c draw starts bottom left
                ((self.position.x + center_point.x) / 2.0) as i32,
                center_point.y as i32,
            ),
            imgproc::FONT_HERSHEY_COMPLEX,
            0.75,
            Scalar::from((255.0, 122.5, 0.0)),
            1,
            LINE_AA,
            false,
        )?;

        Ok(())
    }
}
