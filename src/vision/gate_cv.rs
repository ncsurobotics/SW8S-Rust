use crate::config::ColorProfile;

use super::{image_prep::resize, MatWrapper, PosVector, VisualDetection, VisualDetector, Yuv};
use opencv::{
    core::{in_range, Point, Scalar, Size, Vector},
    imgproc::{
        contour_area_def, cvt_color_def, find_contours_def, min_area_rect, CHAIN_APPROX_SIMPLE,
        COLOR_BGR2YUV, RETR_EXTERNAL,
    },
    prelude::{Mat, MatTraitConst},
};
use std::ops::RangeInclusive;

#[derive(Debug)]
pub struct GateCV {
    color_bounds_red: RangeInclusive<Yuv>,
    color_bounds_black: RangeInclusive<Yuv>,
    size: Size,
    image: MatWrapper,
}

impl GateCV {
    pub fn new(
        color_bounds_red: RangeInclusive<Yuv>,
        color_bounds_black: RangeInclusive<Yuv>,
        size: Size,
    ) -> Self {
        Self {
            color_bounds_red,
            color_bounds_black,
            size,
            image: Mat::default().into(),
        }
    }

    pub fn from_color_profile(color_profile: &ColorProfile) -> Self {
        Self::new(
            color_profile.red.clone(),
            color_profile.black.clone(),
            Size::from((400, 300)),
        )
    }
}

// TODO: Change these to match slalom, not path
impl Default for GateCV {
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
            (Yuv {
                y: 20,
                u: 100,
                v: 160,
            })..=(Yuv {
                y: 220,
                u: 135,
                v: 255,
            }),
            Size::from((400, 300)),
        )
    }
}

impl VisualDetector<f64> for GateCV {
    type ClassEnum = bool;
    type Position = PosVector;

    fn detect(
        &mut self,
        input_image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        self.image = resize(input_image, &self.size)?.into();
        let mut yuv_image = Mat::default();

        cvt_color_def(&self.image.0, &mut yuv_image, COLOR_BGR2YUV)?;

        let red_start = self.color_bounds_red.start();
        let red_end = self.color_bounds_red.end();
        let black_start = self.color_bounds_black.start();
        let black_end = self.color_bounds_black.end();
        let lower_red = Scalar::new(
            red_start.y as f64,
            red_start.u as f64,
            red_start.v as f64,
            0.,
        );
        let upper_red = Scalar::new(red_end.y as f64, red_end.u as f64, red_end.v as f64, 0.);

        let lower_black = Scalar::new(
            black_start.y as f64,
            black_start.u as f64,
            black_start.v as f64,
            0.,
        );
        let upper_black = Scalar::new(
            black_end.y as f64,
            black_end.u as f64,
            black_end.v as f64,
            0.,
        );

        let mut red_mask = Mat::default();
        let _ = in_range(&yuv_image, &lower_red, &upper_red, &mut red_mask);

        let mut black_mask = Mat::default();
        let _ = in_range(&yuv_image, &lower_black, &upper_black, &mut black_mask);

        let mut contours_red = Vector::<Vector<Point>>::new();
        find_contours_def(
            &red_mask,
            &mut contours_red,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
        )?;
        let mut contours_black = Vector::<Vector<Point>>::new();
        find_contours_def(
            &black_mask,
            &mut contours_black,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
        )?;

        let max_contour_red = contours_red.iter().max_by(|x, y| {
            contour_area_def(&x)
                .unwrap()
                .partial_cmp(&contour_area_def(&y).unwrap())
                .unwrap()
        });
        let max_contour_black = contours_black.iter().max_by(|x, y| {
            contour_area_def(&x)
                .unwrap()
                .partial_cmp(&contour_area_def(&y).unwrap())
                .unwrap()
        });

        if let Some(contour_red) = max_contour_red {
            if let Some(contour_black) = max_contour_black {
                let red_rect = min_area_rect(&contour_red).unwrap();
                let black_rect = min_area_rect(&contour_black).unwrap();
                let center_red = red_rect.center;
                let center_black = black_rect.center;

                let red_angle = red_rect.angle;
                let black_angle = black_rect.angle;
                if (red_angle - black_angle).abs() > 20.0 {
                    let red_x = center_red.x;
                    let black_x = center_black.x;
                    let red_y = center_red.y;
                    let black_y = center_black.y;
                    if (red_x - black_x).abs() < 50.0 {
                        let side;
                        if black_y > red_y {
                            // Right
                            side = false;
                        } else {
                            // Left
                            side = true;
                        }
                        let pole_x = (red_x + black_x) / 2.0;
                        let pole_y = (red_y + black_y) / 2.0;
                        return Ok(vec![VisualDetection {
                            class: side,
                            position: PosVector::new(pole_x as f64, pole_y as f64, 0.0, 0.0),
                        }]);
                    }
                }
            }
        }
        Ok(vec![])
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
