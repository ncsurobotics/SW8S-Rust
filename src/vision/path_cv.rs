use super::{image_prep::resize, MatWrapper, PosVector, VisualDetection, VisualDetector, Yuv};
use crate::{config::ColorProfile, vision::Draw};
use opencv::{
    core::{in_range, Point, Scalar, Size, Vector},
    imgproc::{
        self, contour_area_def, cvt_color_def, find_contours_def, min_area_rect,
        CHAIN_APPROX_SIMPLE, COLOR_BGR2YUV, LINE_8, RETR_EXTERNAL,
    },
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};
use std::ops::RangeInclusive;

impl Draw for VisualDetection<bool, PosVector> {
    fn draw(&self, canvas: &mut Mat) -> anyhow::Result<()> {
        let color = if self.class {
            logln!("Drawing true: {:#?}", self.position());
            Scalar::from((0.0, 255.0, 0.0))
        } else {
            Scalar::from((0.0, 0.0, 255.0))
        };

        let angle_rad = (*self.position.angle() as f32) * (3.1415296 / 180.0);
        let b = (angle_rad.cos() * 640.0) / 2.0;
        let a = (angle_rad.sin() * 480.0) / 2.0;

        let start = Point::new(*self.position.x() as i32, *self.position.y() as i32);
        let end = Point::new(
            *self.position.x() as i32 + a as i32,
            *self.position.y() as i32 + b as i32,
        );

        imgproc::circle(
            canvas,
            Point::new(*self.position.x() as i32, *self.position.y() as i32),
            10,
            color,
            2,
            LINE_8,
            0,
        )?;

        imgproc::line(canvas, start, end, color, 2, LINE_8, 0)?;

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

#[derive(Debug)]
pub struct PathCV {
    color_bounds: RangeInclusive<Yuv>,
    size: Size,
    image: MatWrapper,
}

impl PathCV {
    pub fn image(&self) -> Mat {
        (*self.image).clone()
    }
}

impl PathCV {
    pub fn new(color_bounds: RangeInclusive<Yuv>, size: Size) -> Self {
        Self {
            color_bounds,
            size,
            image: Mat::default().into(),
        }
    }

    pub fn from_color_profile(color_profile: &ColorProfile) -> Self {
        Self::new(dbg!(color_profile.orange.clone()), Size::from((400, 300)))
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
            Size::from((400, 300)),
        )
    }
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
        let _ = in_range(&yuv_image, &lower_orange, &upper_orange, &mut mask);

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

                let mut box_rect = Mat::default();
                imgproc::box_points(rect, &mut box_rect)?;

                let box_vec: Vec<Vec<f32>> = box_rect.to_vec_2d()?;

                let zero = box_vec[0].clone();
                let one = box_vec[1].clone();
                let two = box_vec[2].clone();

                let edge1 = (one[0] - zero[0], one[1] - zero[1]);
                let edge2 = (two[0] - one[0], two[1] - one[1]);

                // let longest_edge = max_by_key(edge1, edge2, |e| (e.0.powf(2.0) + e.1.powf(2.0)).sqrt());
                let edge1mag = (edge1.0.powf(2.0) + edge1.1.powf(2.0)).sqrt();
                let edge2mag = (edge2.0.powf(2.0) + edge2.1.powf(2.0)).sqrt();
                let longest_edge = if edge2mag > edge1mag { edge2 } else { edge1 };

                let mut angle = -(longest_edge.0 / longest_edge.1).atan().to_degrees();

                angle = ((angle + 180.0) % 360.0) - 180.0;
                if angle < -90.0 {
                    angle += 180.0;
                }

                println!("{angle:?}");

                let center_adjusted_x = rect.center.x as f64;
                let center_adjusted_y = rect.center.y as f64;

                Ok(vec![VisualDetection {
                    class: true,
                    position: PosVector::new(
                        center_adjusted_x,
                        center_adjusted_y,
                        0.,
                        angle as f64,
                    ),
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
        let _ = in_range(&yuv_image, &lower_orange, &upper_orange, &mut mask);

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

                let mut box_rect = Mat::default();
                imgproc::box_points(rect, &mut box_rect)?;

                let box_vec: Vec<Vec<f32>> = box_rect.to_vec_2d()?;

                let zero = box_vec[0].clone();
                let one = box_vec[1].clone();
                let two = box_vec[2].clone();

                let edge1 = (one[0] - zero[0], one[1] - zero[1]);
                let edge2 = (two[0] - one[0], two[1] - one[1]);

                // let longest_edge = max_by_key(edge1, edge2, |e| (e.0.powf(2.0) + e.1.powf(2.0)).sqrt());
                let edge1mag = (edge1.0.powf(2.0) + edge1.1.powf(2.0)).sqrt();
                let edge2mag = (edge2.0.powf(2.0) + edge2.1.powf(2.0)).sqrt();
                let longest_edge = if edge2mag > edge1mag { edge2 } else { edge1 };

                let mut angle = -(longest_edge.0 / longest_edge.1).atan().to_degrees();

                angle = ((angle + 180.0) % 360.0) - 180.0;
                if angle < -90.0 {
                    angle += 180.0;
                }

                println!("{angle:?}");

                let center_adjusted_x = rect.center.x as f64;
                let center_adjusted_y = rect.center.y as f64;

                Ok(vec![VisualDetection {
                    class: true,
                    position: PosVector::new(
                        center_adjusted_x,
                        center_adjusted_y,
                        0.,
                        angle as f64,
                    ),
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
