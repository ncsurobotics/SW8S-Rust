use std::{fs::create_dir_all, ops::RangeInclusive};

use chrono::Offset;
use itertools::Itertools;
use opencv::{
    core::{in_range, MatTrait, Point, Point2f, Point2i, Rect, Scalar, Size, VecN, Vector, BORDER_DEFAULT, CV_32F},
    imgcodecs::imwrite,
    imgproc::{bounding_rect, contour_area, cvt_color, filter_2d, find_contours, COLOR_RGB2YUV, COLOR_YUV2RGB},
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};
use uuid::Uuid;

use crate::vision::image_prep::{binary_pca, cvt_binary_to_points};

use super::{
    image_prep::resize, pca::PosVector, MatWrapper, Offset2D, VisualDetection, VisualDetector
};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Rgb {
    r: u8,
    g: u8,
    b: u8
}

impl From<&VecN<u8, 3>> for Rgb {
    fn from(value: &VecN<u8, 3>) -> Self {
        Self {
            r: value[0],
            g: value[1],
            b: value[2],
        }
    }
}

impl From<&Rgb> for VecN<u8, 3> {
    fn from(val: &Rgb) -> Self {
        VecN::from_array([val.r, val.g, val.b])
    }
}

impl Into<(u8, u8, u8)> for Rgb {
    fn into(self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }
}

#[derive(Debug)]
pub struct Octagon {
    color_bounds: RangeInclusive<Rgb>,
    size: Size,
    image: MatWrapper,
}

impl Octagon {
    pub fn new(
        color_bounds: RangeInclusive<Rgb>,
        size: Size,
    ) -> Self {
        Self {
            color_bounds,
            size,
            image: Mat::default().into(),
        }
    }
}

impl Default for Octagon {
    fn default() -> Self {
        Octagon::new(
            (Rgb { r: 210, g: 100, b: 210 })..=(Rgb {
                r: 255,
                g: 220,
                b: 255,
            }),
            Size::from((400, 300)),
        )
    }
}

impl VisualDetector<f64> for Octagon {
    type ClassEnum = bool;
    type Position = Offset2D<f64>;

    fn detect(
        &mut self,
        input_image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        let image = resize(input_image, &self.size)?;
        let mut rgb_image = Mat::default();

        cvt_color(&image, &mut rgb_image, COLOR_RGB2YUV, 0).unwrap();
        let image_center = ((rgb_image.cols() / 2) as f64, (rgb_image.rows() / 2) as f64);

        cvt_color(&rgb_image, &mut self.image.0, COLOR_YUV2RGB, 0).unwrap();

        #[cfg(feature = "logging")]
        {
            create_dir_all("/tmp/path_images").unwrap();
            imwrite(
                &("/tmp/path_images/".to_string() + &Uuid::new_v4().to_string() + ".jpeg"),
                &self.image.0,
                &Vector::default(),
            )
            .unwrap();
        }

        let mut mask  = Mat::default();
        // let Rgb { upper_r: , g, b } = self.color_bounds.start();
        let lower: (u8, u8, u8) = self.color_bounds.start().clone().into();
        let upper: (u8, u8, u8) = self.color_bounds.end().clone().into();
        in_range(&rgb_image, &opencv::core::Scalar_::<u8>::from(lower), &opencv::core::Scalar_::<u8>::from(upper), &mut mask).unwrap();
        let mut contours_out = Mat::default();
        find_contours(&mask, &mut contours_out, 3, 2, Point::new(0,0)).unwrap();

        let contour_vec: Vec<Point> = contours_out.iter().unwrap().map(|x| x.1).collect();
        if (contour_vec.len() > 2) {
            let selected_contour = contour_vec.get(contour_vec.len() - 2).unwrap();
            Ok(
                vec![
                    VisualDetection {
                        position: Offset2D::new(selected_contour.x as f64, selected_contour.y as f64),
                        class: true
                    }
                ]
            )

        } else {
            Ok(
                vec![]
            )
        }

    }

    fn normalize(&mut self, pos: &Self::Position) -> Self::Position {
        let img_size = self.image.size().unwrap();
        Self::Position::new(
            ((*pos.x() / (img_size.width as f64)) - 0.5) * 2.0,
            ((*pos.y() / (img_size.height as f64)) - 0.5) * 2.0,
        )
    }
}