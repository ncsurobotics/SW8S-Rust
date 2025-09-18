use super::{image_prep::resize, MatWrapper, Offset2D, VisualDetection, VisualDetector};
use opencv::{
    core::{in_range, Point, Size, VecN, Vector},
    imgproc::{find_contours, CHAIN_APPROX_SIMPLE, RETR_TREE},
    prelude::{Mat, MatTraitConst},
};
use std::ops::RangeInclusive;

#[cfg(feature = "logging")]
use {opencv::imgcodecs::imwrite, std::fs::create_dir_all, uuid::Uuid};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Rgb {
    r: u8,
    g: u8,
    b: u8,
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

impl From<Rgb> for (u8, u8, u8) {
    fn from(val: Rgb) -> Self {
        (val.r, val.g, val.b)
    }
}

impl From<Rgb> for (f64, f64, f64) {
    fn from(val: Rgb) -> Self {
        let (x, y, z): (u8, u8, u8) = val.into();
        (x as f64, y as f64, z as f64)
    }
}

#[derive(Debug)]
pub struct Octagon {
    color_bounds: RangeInclusive<Rgb>,
    size: Size,
    image: MatWrapper,
}

impl Octagon {
    pub fn new(color_bounds: RangeInclusive<Rgb>, size: Size) -> Self {
        Self {
            color_bounds,
            size,
            image: Mat::default().into(),
        }
    }

    pub fn image(&self) -> Mat {
        self.image.0.clone()
    }
}

impl Default for Octagon {
    fn default() -> Self {
        Octagon::new(
            (Rgb {
                r: 210,
                g: 100,
                b: 210,
            })..=(Rgb {
                r: 255,
                g: 180,
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
        self.image.0 = image.clone();

        let mut mask = Mat::default();
        // let Rgb { upper_r: , g, b } = self.color_bounds.start();
        let lower: VecN<u8, 3> = self.color_bounds.start().into();
        let upper: VecN<u8, 3> = self.color_bounds.end().into();
        in_range(&image, &lower, &upper, &mut mask).unwrap();

        #[cfg(feature = "logging")]
        {
            create_dir_all("/tmp/octagon_images").unwrap();
            imwrite(
                &("/tmp/octagon_images/".to_string() + &Uuid::new_v4().to_string() + ".jpeg"),
                &image,
                &Vector::default(),
            )
            .unwrap();
        }

        #[cfg(feature = "logging")]
        {
            create_dir_all("/tmp/masks").unwrap();
            imwrite(
                &("/tmp/masks/".to_string() + &Uuid::new_v4().to_string() + ".jpeg"),
                &mask,
                &Vector::default(),
            )
            .unwrap();
        }

        println!("MASK: {mask:#?}");

        let mut contours_out: Vector<Vector<Point>> = Vector::new();
        find_contours(
            &mask,
            &mut contours_out,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )
        .unwrap();

        if contours_out.len() > 2 {
            let selected_contour_set = contours_out.get(contours_out.len() - 2).unwrap();
            Ok(selected_contour_set
                .iter()
                .map(|contour| VisualDetection {
                    position: Offset2D::new(contour.x as f64, contour.y as f64),
                    class: true,
                })
                .collect())
        } else {
            Ok(vec![])
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
