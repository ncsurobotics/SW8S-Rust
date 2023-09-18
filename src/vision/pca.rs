use anyhow::Result;
use opencv::{
    core::{Point, Scalar},
    imgproc::{self, LINE_8},
    prelude::Mat,
};

use super::{Angle2D, Draw, RelPos, RelPosAngle};

#[derive(Debug, Clone)]
pub struct PosVector {
    x: f64,
    y: f64,
    angle: f64,
    width: f64,
}

impl PosVector {
    pub fn new(x: f64, y: f64, angle: f64, width: f64) -> Self {
        Self { x, y, angle, width }
    }
}

impl Draw for PosVector {
    fn draw(&self, canvas: &mut Mat) -> Result<()> {
        println!("PosVector: {:?}", self);
        imgproc::circle(
            canvas,
            Point::new(self.x as i32, self.y as i32 + 300),
            10,
            Scalar::from((0.0, 255.0, 0.0)),
            2,
            LINE_8,
            0,
        )?;
        Ok(())
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
