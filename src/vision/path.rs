use std::ops::{Range, RangeBounds, RangeInclusive};

use itertools::Itertools;
use opencv::{
    core::{
        in_range, KmeansFlags, MatIter, Mat_, Scalar, Size, TermCriteria, VecN, Vector, CV_32FC1,
        CV_32FC3, CV_8UC3,
    },
    imgcodecs::imwrite,
    imgproc::{circle, cvt_color, COLOR_RGB2YUV, LINE_8},
    prelude::{DataType, Mat, MatTraitConst, MatTraitConstManual},
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
        }
    }
}

impl Default for Path {
    fn default() -> Self {
        Path::new(
            (Yuv { y: 0, u: 0, v: 152 })..=(Yuv {
                y: 255,
                u: 152,
                v: 255,
            }),
            20.0..=800.0,
            4,
            Size::from((400, 300)),
            0,
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
        image: &Mat,
    ) -> anyhow::Result<Vec<VisualDetection<Self::ClassEnum, Self::Position>>> {
        let original_size = image.size().unwrap();
        let mut image = resize(image, &self.size)?;
        imwrite("resized.jpeg", &image, &Vector::default()).unwrap();
        let mut yuv_image = Mat::default();

        let post_kmeans = kmeans(&image);
        imwrite("test.jpeg", &post_kmeans, &Vector::default()).unwrap();
        cvt_color(&post_kmeans, &mut yuv_image, COLOR_RGB2YUV, 0).unwrap();
        /*
        let mut post_kmeans = Mat::default();
        let mut yuv_image = Mat::default();

        let original_rows = image.rows();
        let original_channels = image.channels();
        let image = image.reshape(3, 0).unwrap();
        println!("{:?}", image.size().unwrap());
        let mut converted_image = Mat::default();
        image.convert_to(&mut converted_image, CV_32FC3, 1.0 / 255.0, 0.0)?;

        //let unshaped = image.reshape(original_shape)?;
        //imwrite("reshaped.jpeg", &unshaped, &Vector::default()).unwrap();
        //let mut reconv = Mat::default();

        println!("POS 9");
        kmeans(
            &converted_image,
            self.num_regions,
            &mut post_kmeans,
            TermCriteria::default().unwrap(),
            self.attempts,
            KmeansFlags::KMEANS_PP_CENTERS as i32,
            &mut Mat::default(),
        )?;
        println!("POS 1");
        let mut post_kmeans_converted = Mat::default();
        //post_kmeans
        //    .convert_to(&mut post_kmeans_converted, CV_8UC3, 255.0, 0.0)
        //    .unwrap();
        //let post_kmeans = post_kmeans.reshape(original_channels, 0).unwrap();
        let mut post_kmeans_converted = Mat::default();
        post_kmeans
            .convert_to(&mut post_kmeans_converted, CV_8UC3, 255.0, 0.0)
            .unwrap();
        let post_kmeans_converted = post_kmeans_converted.reshape(original_channels, 0).unwrap();
        imwrite("test.jpeg", &post_kmeans_converted, &Vector::default()).unwrap();
        cvt_color(&post_kmeans_converted, &mut yuv_image, COLOR_RGB2YUV, 0).unwrap();

        */
        let image_center = (
            (post_kmeans.cols() / 2) as f64,
            (post_kmeans.rows() / 2) as f64,
        );

        yuv_image
            .iter::<VecN<u8, 3>>()
            .unwrap()
            .sorted_by(|(_, val), (_, n_val)| Ord::cmp(val.as_slice(), n_val.as_slice()))
            .dedup_by(|(_, val), (_, n_val)| val == n_val)
            .map(|(_, val)| {
                let mut bin_image = Mat::default();
                in_range(&yuv_image, &val, &val, &mut bin_image).unwrap();
                let on_points = cvt_binary_to_points(&bin_image.try_into_typed().unwrap());

                on_points.iter().for_each(|p| {
                    circle(
                        &mut image,
                        opencv::core::Point_ {
                            x: p.x as i32,
                            y: p.y as i32,
                        },
                        5,
                        Scalar::from((0.0, 255.0, 0.0)),
                        -1,
                        LINE_8,
                        0,
                    )
                    .unwrap()
                });
                println!("Points len: {}", on_points.len());
                imwrite(
                    "tests/vision/output/path_images/1.jpeg",
                    &image,
                    &Vector::default(),
                );

                let pca_output = binary_pca(&on_points, 0).unwrap();

                println!("PCA Output: {:#?}", pca_output);
                //println!("PCA Values: {:#?}", pca_output.pca_value());

                let (length_idx, width_idx) = if pca_output.pca_value().get(1).unwrap()
                    > pca_output.pca_value().get(0).unwrap()
                {
                    (1, 0)
                } else {
                    (0, 1)
                };
                let width = pca_output.pca_value().get(width_idx).unwrap();

                let valid = self.width_bounds.contains(&width)
                    && Yuv::from(&val).in_range(&self.color_bounds);

                let p_vec = PosVector::new(
                    (pca_output.mean().get(0).unwrap())
                        * (original_size.width / self.size.width) as f64,
                    (pca_output.mean().get(1).unwrap())
                        * (original_size.height / self.size.height) as f64,
                    compute_angle(
                        (
                            pca_output.pca_vector().get(length_idx).unwrap(),
                            pca_output.pca_vector().get(length_idx + 1).unwrap(),
                        ),
                        FORWARD,
                    ),
                    width,
                );

                Ok(VisualDetection {
                    class: valid,
                    confidence: 1.0,
                    position: p_vec,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use opencv::{
        core::Vector,
        imgcodecs::{imread, imwrite, IMREAD_COLOR},
    };

    use crate::vision::Draw;

    use super::*;

    #[test]
    fn detect_single() {
        let mut image = imread("tests/vision/resources/path_images/1.jpeg", IMREAD_COLOR).unwrap();
        let detections = Path::default().detect(&image).unwrap();

        detections
            .iter()
            .for_each(|result| result.position().draw(&mut image).unwrap());

        println!("Detections: {:#?}", detections);
        panic!();
        imwrite(
            "tests/vision/output/path_images/1.jpeg",
            &image,
            &Vector::default(),
        )
        .unwrap();
    }
}
