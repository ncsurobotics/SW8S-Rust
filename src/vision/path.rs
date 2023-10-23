use std::ops::RangeInclusive;

use itertools::Itertools;
use opencv::{
    core::{in_range, Size, VecN},
    imgproc::{cvt_color, COLOR_RGB2YUV, COLOR_YUV2RGB},
    prelude::{Mat, MatTraitConst},
};

use crate::vision::image_prep::{binary_pca, cvt_binary_to_points};

use super::{
    image_prep::{kmeans, resize},
    pca::PosVector,
    VisualDetection, VisualDetector,
};

static FORWARD: (f64, f64) = (0.0, -1.0);

#[derive(Debug, PartialEq)]
pub struct Yuv {
    y: u8,
    u: u8,
    v: u8,
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
pub struct Path {
    color_bounds: RangeInclusive<Yuv>,
    width_bounds: RangeInclusive<f64>,
    num_regions: i32,
    size: Size,
    attempts: i32,
    image: Mat,
}

impl Path {
    pub fn image(&self) -> &Mat {
        &self.image
    }
}

impl Path {
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
            image: Mat::default(),
        }
    }
}

impl Default for Path {
    fn default() -> Self {
        Path::new(
            (Yuv { y: 0, u: 0, v: 127 })..=(Yuv {
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

impl VisualDetector<i32> for Path {
    type ClassEnum = bool;
    type Position = PosVector;

    fn detect(
        &mut self,
        input_image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        let image = resize(input_image, &self.size)?;
        let mut yuv_image = Mat::default();

        cvt_color(&image, &mut yuv_image, COLOR_RGB2YUV, 0).unwrap();
        yuv_image = kmeans(&yuv_image, self.num_regions, self.attempts);
        let image_center = ((yuv_image.cols() / 2) as f64, (yuv_image.rows() / 2) as f64);

        cvt_color(&yuv_image, &mut self.image, COLOR_YUV2RGB, 0).unwrap();

        yuv_image
            .iter::<VecN<u8, 3>>()
            .unwrap()
            .sorted_by(|(_, val), (_, n_val)| Ord::cmp(val.as_slice(), n_val.as_slice()))
            .dedup_by(|(_, val), (_, n_val)| val == n_val)
            .map(|(_, val)| {
                let mut bin_image = Mat::default();
                in_range(&yuv_image, &val, &val, &mut bin_image).unwrap();
                let on_points = cvt_binary_to_points(&bin_image.try_into_typed().unwrap());
                let pca_output = binary_pca(&on_points, 0).unwrap();

                let (length_idx, width_idx) = if pca_output.pca_value().get(1).unwrap()
                    > pca_output.pca_value().get(0).unwrap()
                {
                    (1, 0)
                } else {
                    (0, 1)
                };
                // width bounds have a temp fix -- not sure why output is so large
                let width = pca_output.pca_value().get(width_idx).unwrap() / 100.0;
                let length = pca_output.pca_value().get(length_idx).unwrap();
                let length_2 = pca_output.pca_vector().get(length_idx + 1).unwrap();

                println!("Testing for valid...");
                println!("\tself.width_bounds = {:?}", self.width_bounds);
                println!("\tself.width = {:?}", width);
                println!(
                    "\tcontained_width = {:?}",
                    self.width_bounds.contains(&width)
                );
                println!();
                println!("\tYUV range = {:?}", self.color_bounds);
                println!("\tYUV val = {:?}", Yuv::from(&val));
                println!(
                    "\tcontained_color = {:?}",
                    Yuv::from(&val).in_range(&self.color_bounds)
                );
                println!();

                let valid = self.width_bounds.contains(&width)
                    && Yuv::from(&val).in_range(&self.color_bounds);

                let p_vec = PosVector::new(
                    ((pca_output.mean().get(0).unwrap()) - image_center.0)
                        + (self.image.size().unwrap().width as f64) / 2.0,
                    (pca_output.mean().get(1).unwrap()) - image_center.1
                        + (self.image.size().unwrap().height as f64) / 2.0,
                    compute_angle(
                        (
                            pca_output.pca_vector().get(length_idx).unwrap(),
                            pca_output.pca_vector().get(length_idx + 1).unwrap(),
                        ),
                        FORWARD,
                    ),
                    width,
                    length / 300.0,
                    length_2,
                );

                Ok(VisualDetection {
                    class: valid,
                    position: p_vec,
                })
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use opencv::{
        core::Vector,
        imgcodecs::{imread, imwrite, IMREAD_COLOR},
    };

    use crate::vision::Draw;

    use super::*;

    #[test]
    fn detect_single() {
        let image = imread("tests/vision/resources/path_images/1.jpeg", IMREAD_COLOR).unwrap();
        let mut path = Path::default();
        let detections = path.detect(&image).unwrap();
        let mut shrunk_image = path.image().clone();

        detections
            .iter()
            .for_each(|result| result.draw(&mut shrunk_image).unwrap());

        println!("Detections: {:#?}", detections);

        create_dir_all("tests/vision/output/path_images").unwrap();
        imwrite(
            "tests/vision/output/path_images/1.jpeg",
            &shrunk_image,
            &Vector::default(),
        )
        .unwrap();
    }
}
