use crate::config::ColorProfile;

use super::{image_prep::resize, MatWrapper, PosVector, VisualDetection, VisualDetector, Yuv};
use opencv::{
    core::{in_range, Point, Scalar, Size, Vector},
    imgproc::{
        box_points, contour_area_def, cvt_color_def, find_contours_def, min_area_rect,
        CHAIN_APPROX_SIMPLE, COLOR_BGR2YUV, RETR_EXTERNAL,
    },
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};
use std::ops::RangeInclusive;

#[derive(Debug)]
pub struct Slalom {
    color_bounds: RangeInclusive<Yuv>,
    area_bounds: RangeInclusive<f64>,
    size: Size,
    image: MatWrapper,
}

impl Slalom {
    pub fn new(
        color_bounds: RangeInclusive<Yuv>,
        area_bounds: RangeInclusive<f64>,
        size: Size,
    ) -> Self {
        Self {
            color_bounds,
            area_bounds,
            size,
            image: Mat::default().into(),
        }
    }

    pub fn from_color_profile(
        color_profile: &ColorProfile,
        area_bounds: RangeInclusive<f64>,
    ) -> Self {
        Self::new(
            color_profile.red.clone(),
            area_bounds,
            Size::from((400, 300)),
        )
    }
}

// TODO: Change these to match slalom, not path
impl Default for Slalom {
    fn default() -> Self {
        Self::new(
            (Yuv {
                y: 20,
                u: 100,
                v: 160,
            })..=(Yuv {
                y: 220,
                u: 135,
                v: 255,
            }),
            1000.0..=11000.0,
            Size::from((400, 300)),
        )
    }
}

impl VisualDetector<f64> for Slalom {
    type ClassEnum = bool;
    type Position = PosVector;

    fn detect(
        &mut self,
        input_image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        let areas = self.area_bounds.clone();
        let min_area = areas.start();
        let max_area = areas.end();

        self.image = resize(input_image, &self.size)?.into();
        let mut yuv_image = Mat::default();

        cvt_color_def(&self.image.0, &mut yuv_image, COLOR_BGR2YUV)?;

        let color_start = self.color_bounds.start();
        let color_end = self.color_bounds.end();
        let lower_red = Scalar::new(
            color_start.y as f64,
            color_start.u as f64,
            color_start.v as f64,
            0.,
        );
        let upper_red = Scalar::new(
            color_end.y as f64,
            color_end.u as f64,
            color_end.v as f64,
            0.,
        );

        let mut mask = Mat::default();
        let _ = in_range(&yuv_image, &lower_red, &upper_red, &mut mask);

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
            #[cfg(feature = "logging")]
            logln!("AREA: {area}");

            if area > *min_area && area < *max_area {
                let rect = min_area_rect(&contour)?;

                let mut box_rect = Mat::default();
                box_points(rect, &mut box_rect)?;

                let box_vec: Vec<Vec<f32>> = box_rect.to_vec_2d()?;

                let zero = box_vec[0].clone();
                let one = box_vec[1].clone();
                let two = box_vec[2].clone();

                let edge1 = (one[0] - zero[0], one[1] - zero[1]);
                let edge2 = (two[0] - one[0], two[1] - one[1]);

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
