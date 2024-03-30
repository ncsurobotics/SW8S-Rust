use derive_getters::Getters;
use opencv::{
    core::{Point, Scalar},
    imgproc::{self, LINE_8},
    prelude::Mat,
};

use super::{Angle2D, Draw, RelPosAngle, VisualDetection};

#[derive(Debug, Clone, Getters)]
pub struct PosVector {
    x: f64,
    y: f64,
    angle: f64,
    width: f64,
    length: f64,
    length_2: f64,
}

impl PosVector {
    pub fn new(x: f64, y: f64, angle: f64, width: f64, length: f64, length_2: f64) -> Self {
        Self {
            x,
            y,
            angle,
            width,
            length,
            length_2,
        }
    }
}

impl RelPosAngle for PosVector {
    type Number = f64;

    fn offset_angle(&self) -> Angle2D<Self::Number> {
        Angle2D {
            x: self.x,
            y: self.y,
            angle: self.angle,
        }
    }
}

impl Draw for VisualDetection<bool, PosVector> {
    fn draw(&self, canvas: &mut Mat) -> anyhow::Result<()> {
        let color = if self.class {
            Scalar::from((0.0, 255.0, 0.0))
        } else {
            Scalar::from((0.0, 0.0, 255.0))
        };

        imgproc::circle(
            canvas,
            Point::new(*self.position.x() as i32, *self.position.y() as i32),
            10,
            color,
            2,
            LINE_8,
            0,
        )?;

        imgproc::arrowed_line(
            canvas,
            Point::new(*self.position.x() as i32, *self.position.y() as i32),
            Point::new(
                (self.position.x() + 0.02 * self.position.length() * self.position.length()) as i32,
                (self.position.y() + 0.4 * self.position.length_2() * self.position.length())
                    as i32,
            ),
            color,
            2,
            LINE_8,
            0,
            0.1,
        )?;
        Ok(())
    }
}

impl Draw for PosVector {
    fn draw(&self, canvas: &mut Mat) -> anyhow::Result<()> {
        let color = Scalar::from((0.0, 255.0, 0.0));

        imgproc::circle(
            canvas,
            Point::new(*self.x() as i32, *self.y() as i32),
            10,
            color,
            2,
            LINE_8,
            0,
        )?;

        imgproc::arrowed_line(
            canvas,
            Point::new(*self.x() as i32, *self.y() as i32),
            Point::new(
                (self.x() + 0.02 * self.length() * self.length()) as i32,
                (self.y() + 0.4 * self.length_2() * self.length()) as i32,
            ),
            color,
            2,
            LINE_8,
            0,
            0.1,
        )?;
        Ok(())
    }
}
