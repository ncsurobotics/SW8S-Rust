/*
use std::{fs::create_dir_all, ops::RangeInclusive};

use itertools::Itertools;
use opencv::{
    core::{in_range, MatTrait, Point, Point2f, Point2i, Size, VecN, Vector, BORDER_DEFAULT, CV_32F},
    imgcodecs::imwrite,
    imgproc::{cvt_color, filter_2d, find_contours, COLOR_RGB2YUV, COLOR_YUV2RGB},
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};
use uuid::Uuid;

use crate::vision::image_prep::{binary_pca, cvt_binary_to_points};

use super::{
    image_prep::{resize},
    pca::PosVector,
    MatWrapper, VisualDetection, VisualDetector,
};

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub struct Octagon {
    color_bounds: RangeInclusive<Rgb>,
    width_bounds: RangeInclusive<f64>,
    num_regions: i32,
    size: Size,
    attempts: i32,
    image: MatWrapper,
}

impl Octagon {
    pub fn new(
        color_bounds: RangeInclusive<Rgb>,
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

impl Default for Octagon {
    fn default() -> Self {
        Octagon::new(
            (Rgb { r: 210, g: 100, b: 210 })..=(Rgb {
                r: 255,
                g: 220,
                b: 255,
            }),
            20.0..=800.0,
            4,
            Size::from((400, 300)),
            3,
        )
    }
}

impl VisualDetector<f64> for Octagon {
    type ClassEnum = bool;
    type Position = PosVector;

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

        let kernel = (Mat::ones_nd(&[2,2], CV_32F).unwrap() / 4.0).into_result().unwrap();
        let mut kernel_out = Mat::default();
        kernel_out.set_matexpr(&kernel);
        let mut rgb_filtered_out = Mat::default();
        filter_2d(&rgb_image, &mut rgb_filtered_out, -1, &kernel_out, Point::new(-1, -1), 0.0, BORDER_DEFAULT);

        rgb_filtered_out
        .iter::<VecN<u8, 3>>()
        .unwrap()
        .sorted_by(|(_, val), (_, n_val)| Ord::cmp(val.as_slice(), n_val.as_slice()))
        .dedup_by(|(_, val), (_, n_val)| val == n_val)
        .map(|(_, val)| {
            let mut bin_image = Mat::default();
            in_range(&rgb_filtered_out, &val, &val, &mut bin_image).unwrap();

            // Ok(VisualDetection {
            //     class: valid,
            //     position: p_vec,
            // })
        })
        .collect()
        // let mut contours_out: VecN<opencv::core::Point2f, 0> = VecN::default();
        // find_contours(&rgb_filtered_out, &mut contours_out, 3, 2, Point::new(0,0));
        // contours_out.map(|v: Point2f| {

        // });

    }

    fn normalize(&mut self, pos: &Self::Position) -> Self::Position {
        let img_size = self.image.size().unwrap();
        Self::Position::new(
            ((*pos.x() / (img_size.width as f64)) - 0.5) * 2.0,
            ((*pos.y() / (img_size.height as f64)) - 0.5) * 2.0,
            *pos.angle(),
            *pos.width() / (img_size.width as f64),
            *pos.length() / (img_size.height as f64),
            *pos.length_2() / (img_size.height as f64),
        )
    }
}
*/
