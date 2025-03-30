use std::{fs::create_dir_all, ops::Mul, ops::RangeInclusive};

use derive_getters::Getters;
use itertools::Itertools;
use opencv::{
    core::{in_range, Point, Scalar, Size, VecN, Vector},
    imgcodecs::imwrite,
    imgproc::{
        self, box_points, circle, contour_area_def, cvt_color_def, find_contours_def,
        min_area_rect, CHAIN_APPROX_SIMPLE, COLOR_BGR2YUV, COLOR_YUV2BGR, LINE_8, RETR_EXTERNAL,
    },
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};
use uuid::Uuid;

use crate::vision::image_prep::{binary_pca, cvt_binary_to_points};
use crate::vision::{Angle2D, Draw, Offset2D, RelPosAngle};

use super::{
    image_prep::{kmeans, resize},
    MatWrapper, VisualDetection, VisualDetector,
};

static FORWARD: (f64, f64) = (0.0, -1.0);

#[derive(Debug, Clone, Getters, PartialEq)]
pub struct PosVector {
    x: f64,
    y: f64,
    z: f64,
    angle: f64,
}

impl PosVector {
    fn new(x: f64, y: f64, z: f64, angle: f64) -> Self {
        Self { x, y, z, angle }
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

impl Mul<&Mat> for PosVector {
    type Output = Self;

    fn mul(self, rhs: &Mat) -> Self::Output {
        let size = rhs.size().unwrap();
        Self {
            x: (self.x + 0.5) * (size.width as f64),
            y: (self.y + 0.5) * (size.height as f64),
            z: 0.,
            angle: self.angle,
        }
    }
}

impl Draw for VisualDetection<bool, PosVector> {
    fn draw(&self, canvas: &mut Mat) -> anyhow::Result<()> {
        let color = if self.class {
            logln!("Drawing true: {:#?}", self.position());
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

        // imgproc::arrowed_line(
        //     canvas,
        //     Point::new(*self.position.x() as i32, *self.position.y() as i32),
        //     Point::new(
        //         (self.position.x() + 0.02 * self.position.length() * self.position.length()) as i32,
        //         (self.position.y() + 0.4 * self.position.length_2() * self.position.length())
        //             as i32,
        //     ),
        //     color,
        //     2,
        //     LINE_8,
        //     0,
        //     0.1,
        // )?;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Yuv {
    pub y: u8,
    pub u: u8,
    pub v: u8,
}

impl From<&VecN<u8, 3>> for Yuv {
    fn from(value: &VecN<u8, 3>) -> Self {
        Self {
            y: value[0],
            u: value[1],
            v: value[2],
        }
    }
}

impl From<&Yuv> for VecN<u8, 3> {
    fn from(val: &Yuv) -> Self {
        VecN::from_array([val.y, val.u, val.v])
    }
}

impl Yuv {
    fn in_range(&self, range: &RangeInclusive<Self>) -> bool {
        self.y >= range.start().y
            && self.u >= range.start().u
            && self.v >= range.start().v
            && self.y <= range.end().y
            && self.u <= range.end().u
            && self.v <= range.end().v
    }
}

#[derive(Debug)]
pub struct PathCV {
    color_bounds: RangeInclusive<Yuv>,
    width_bounds: RangeInclusive<f64>,
    num_regions: i32,
    size: Size,
    attempts: i32,
    image: MatWrapper,
}

impl PathCV {
    pub fn image(&self) -> Mat {
        (*self.image).clone()
    }
}

impl PathCV {
    pub fn new(
        color_bounds: RangeInclusive<Yuv>,
        width_bounds: RangeInclusive<f64>,
        num_regions: i32,
        size: Size,
        attempts: i32,
    ) -> Self {
        Self {
            color_bounds,
            width_bounds,
            num_regions,
            size,
            attempts,
            image: Mat::default().into(),
        }
    }
}

impl Default for PathCV {
    fn default() -> Self {
        PathCV::new(
            (Yuv { y: 0, u: 0, v: 175 })..=(Yuv {
                y: 255,
                u: 127,
                v: 255,
            }),
            20.0..=800.0,
            4,
            Size::from((400, 300)),
            3,
        )
    }
}

fn compute_angle(v1: (f64, f64), v2: (f64, f64)) -> f64 {
    let dot = (v1.0 * v2.0) + (v1.1 * v2.1);
    let norm = |vec: (f64, f64)| ((vec.0 * vec.0) + (vec.1 * vec.1)).sqrt();
    let norm_combined = norm(v1) * norm(v2);
    (dot / norm_combined).acos()
}

impl VisualDetector<i32> for PathCV {
    type ClassEnum = bool;
    type Position = PosVector;

    fn detect(
        &mut self,
        input_image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        self.image = resize(input_image, &self.size)?.into();
        let mut yuv_image = Mat::default();

        cvt_color_def(&self.image.0, &mut yuv_image, COLOR_BGR2YUV)?;

        let color_start = self.color_bounds.start();
        let color_end = self.color_bounds.end();
        let lower_orange = Scalar::new(
            color_start.y as f64,
            color_start.u as f64,
            color_start.v as f64,
            0.,
        );
        let upper_orange = Scalar::new(
            color_end.y as f64,
            color_end.u as f64,
            color_end.v as f64,
            0.,
        );

        let mut mask = Mat::default();
        in_range(&yuv_image, &lower_orange, &upper_orange, &mut mask);

        let mut contours = Vector::<Vector<Point>>::new();
        find_contours_def(&mask, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)?;

        let max_contour = contours.iter().max_by(|x, y| {
            contour_area_def(&x)
                .unwrap()
                .partial_cmp(&contour_area_def(&y).unwrap())
                .unwrap()
        });

        if let Some(contour) = max_contour {
            let area = contour_area_def(&contour)?;
            if area > 5000.0 {
                let rect = min_area_rect(&contour)?;
                let mut angle = rect.angle as f64;
                let width = rect.size.width;
                let height = rect.size.height;
                if width > height {
                    angle = rect.angle as f64;
                } else {
                    angle = (rect.angle as f64) + 90.0;
                }
                angle -= 90.0;
                let mut box_rect = Mat::default();
                box_points(rect, &mut box_rect)?;

                println!("{:?}", angle);
                // DRAW STUFF
                // SEND TO RTSP SERVER

                // let center_adjusted_x =
                //     (rect.center.x as f64) - ((self.image.size()?.width as f64) / 2.0);
                // let center_adjusted_y =
                //     ((self.image.size()?.height as f64) / 2.0) - (rect.center.y as f64);
                let center_adjusted_x = rect.center.x as f64;
                let center_adjusted_y = rect.center.y as f64;

                Ok(vec![VisualDetection {
                    class: true,
                    position: PosVector::new(center_adjusted_x, center_adjusted_y, 0., angle),
                }])
            } else {
                Ok(vec![VisualDetection {
                    class: false,
                    position: PosVector::new(0., 0., 0., 0.),
                }])
            }
        } else {
            Ok(vec![VisualDetection {
                class: false,
                position: PosVector::new(0., 0., 0., 0.),
            }])
        }
    }

    fn normalize(&mut self, pos: &Self::Position) -> Self::Position {
        let img_size = self.image.size().unwrap();
        Self::Position::new(
            ((*pos.x() / (img_size.width as f64)) - 0.5) * 2.0,
            ((*pos.y() / (img_size.height as f64)) - 0.5) * 2.0,
            0.,
            *pos.angle(),
        )
    }
}

impl VisualDetector<f64> for PathCV {
    type ClassEnum = bool;
    type Position = PosVector;

    fn detect(
        &mut self,
        input_image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        self.image = resize(input_image, &self.size)?.into();
        let mut yuv_image = Mat::default();

        cvt_color_def(&self.image.0, &mut yuv_image, COLOR_BGR2YUV)?;

        let color_start = self.color_bounds.start();
        let color_end = self.color_bounds.end();
        let lower_orange = Scalar::new(
            color_start.y as f64,
            color_start.u as f64,
            color_start.v as f64,
            0.,
        );
        let upper_orange = Scalar::new(
            color_end.y as f64,
            color_end.u as f64,
            color_end.v as f64,
            0.,
        );

        let mut mask = Mat::default();
        in_range(&yuv_image, &lower_orange, &upper_orange, &mut mask);

        let mut contours = Vector::<Vector<Point>>::new();
        find_contours_def(&mask, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)?;

        let max_contour = contours.iter().max_by(|x, y| {
            contour_area_def(&x)
                .unwrap()
                .partial_cmp(&contour_area_def(&y).unwrap())
                .unwrap()
        });

        if let Some(contour) = max_contour {
            let area = contour_area_def(&contour)?;
            if area > 500.0 {
                let rect = min_area_rect(&contour)?;
                let mut angle = rect.angle as f64;
                let width = rect.size.width;
                let height = rect.size.height;
                if width > height {
                    angle = rect.angle as f64;
                } else {
                    angle = (rect.angle as f64) + 90.0;
                }
                angle -= 90.0;
                let mut box_rect = Mat::default();
                box_points(rect, &mut box_rect)?;

                println!("{:?}", angle);
                // DRAW STUFF
                // SEND TO RTSP SERVER

                // let center_adjusted_x =
                //     (rect.center.x as f64) - ((self.image.size()?.width as f64) / 2.0);
                // let center_adjusted_y =
                //     ((self.image.size()?.height as f64) / 2.0) - (rect.center.y as f64);
                let center_adjusted_x = rect.center.x as f64;
                let center_adjusted_y = rect.center.y as f64;

                Ok(vec![VisualDetection {
                    class: true,
                    position: PosVector::new(center_adjusted_x, center_adjusted_y, 0., angle),
                }])
            } else {
                Ok(vec![VisualDetection {
                    class: false,
                    position: PosVector::new(0., 0., 0., 0.),
                }])
            }
        } else {
            Ok(vec![VisualDetection {
                class: false,
                position: PosVector::new(0., 0., 0., 0.),
            }])
        }
    }

    fn normalize(&mut self, pos: &Self::Position) -> Self::Position {
        let img_size = self.image.size().unwrap();
        Self::Position::new(
            ((*pos.x() / (img_size.width as f64)) - 0.5) * 2.0,
            ((*pos.y() / (img_size.height as f64)) - 0.5) * 2.0,
            0.,
            *pos.angle(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use opencv::{
        core::Vector,
        imgcodecs::{imread, imwrite, IMREAD_COLOR},
    };

    use crate::{logln, vision::Draw};

    use super::*;

    #[test]
    fn detect_single() {
        let image = imread("tests/vision/resources/path_images/1.jpeg", IMREAD_COLOR).unwrap();
        let mut path = PathCV::default();
        let detections = <PathCV as VisualDetector<f64>>::detect(&mut path, &image).unwrap();
        let mut shrunk_image = path.image().clone();

        detections.iter().for_each(|result| {
            <VisualDetection<_, _> as Draw>::draw(result, &mut shrunk_image).unwrap()
        });

        logln!("Detections: {:#?}", detections);

        create_dir_all("tests/vision/output/path_images").unwrap();
        imwrite(
            "tests/vision/output/path_images/1.jpeg",
            &shrunk_image,
            &Vector::default(),
        )
        .unwrap();
    }
}
