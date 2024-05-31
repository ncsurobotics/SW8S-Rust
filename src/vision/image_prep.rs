use std::hash::Hash;
use std::ops::Deref;

use itertools::Itertools;
use opencv::{
    core::{
        pca_compute2, DataType, Mat_, Point_, Scalar, Size, TermCriteria, VecN, Vector, CMP_EQ,
        CV_32F, CV_32FC3, CV_64F, CV_8U, KMEANS_PP_CENTERS, ROTATE_90_COUNTERCLOCKWISE,
    },
    imgproc::{self},
    prelude::{Mat, MatSizeTraitConst, MatTrait, MatTraitConst, MatTraitConstManual},
};

use anyhow::Result;

/// Creates a new Mat with the specified size
///
/// # Arguments
/// * `frame` - Original matrix
/// * `target_size` - Dimensions for output matrix
///
/// # Examples
/// ```
/// use opencv::{
///     core::{Mat, Size},
///     prelude::{MatTraitConst, MatTraitConstManual, MatSizeTraitConst},
/// };
/// use sw8s_rust_lib::vision::image_prep::resize;
///
/// let raw_mat: [u8; 7] = [0, 5, 32, 32, 5, 0, 1];
/// let mat = Mat::from_slice(&raw_mat).unwrap().clone_pointee();
///
/// assert_eq!(resize(&mat, &Size::new(2, 2)).unwrap().mat_size().apply().unwrap(), Size::new(2, 2));
/// ```
pub fn resize(frame: &Mat, target_size: &Size) -> Result<Mat> {
    let mut res = Mat::default();
    imgproc::resize(
        frame,
        &mut res,
        *target_size,
        0.0,
        0.0,
        3, // InterpolationFlags::INTER_AREA,
    )?;
    Ok(res)
}

/// Returns true if the image size is within the bounds
///
/// # Arguments
/// * `image` - Mat under test
/// * `bounds` - Maximum allowed size for `image`
///
/// # Examples
/// ```
/// use opencv::core::{Mat, Size};
/// use sw8s_rust_lib::vision::image_prep::check_bounds;
///
/// let raw_mat: [u8; 7] = [0, 0, 0, 255, 255, 0, 0];
/// let mat = Mat::from_slice(&raw_mat).unwrap().clone_pointee();
///
/// assert!(check_bounds(&mat, &Size::new(7, 1)));
/// assert!(check_bounds(&mat, &Size::new(11, 2)));
///
/// assert!(!check_bounds(&mat, &Size::new(5, 1)));
/// assert!(!check_bounds(&mat, &Size::new(5, 2)));
/// ```
pub fn check_bounds(image: &Mat, bounds: &Size) -> bool {
    let image_sizes = image
        .mat_size()
        .apply()
        .expect("Image provided to check_bounds doesn't have a valid size");
    (image_sizes.width <= bounds.width) && (image_sizes.height <= bounds.height)
}

/// Returns the dimensions for evenly divided blocks
///
/// # Arguments
/// * `image` - Image to count block sizes from
/// * `num_x` - Number of blocks in X dimension
/// * `num_y` - Number of blocks in Y dimension
///
/// # Examples
/// ```
/// use opencv::{
///     core::{Mat, Size},
///     prelude::{MatTraitConst, MatTraitConstManual, MatSizeTraitConst},
/// };
/// use sw8s_rust_lib::vision::image_prep::slice_number;
///
/// let raw_mat: [u8; 8] = [0, 5, 32, 32, 5, 0, 1, 3];
/// let mat = Mat::from_slice(&raw_mat).unwrap().clone_pointee();
///
/// assert_eq!(slice_number(&mat, 2, 1).unwrap(), Size::new(4, 1));
/// ```
pub fn slice_number(image: &Mat, num_x: i32, num_y: i32) -> Result<Size> {
    let image_sizes = image.mat_size().apply()?;
    Ok(Size::new(
        image_sizes.width / num_x,
        image_sizes.height / num_y,
    ))
}

/// Read-only struct for results from PCA computation
#[derive(Debug, Default)]
pub struct PcaData {
    mean: Vector<f64>,
    pca_vector: Vector<f64>,
    pca_value: Vector<f64>,
}

impl PcaData {
    pub fn mean(&self) -> &Vector<f64> {
        &self.mean
    }

    pub fn pca_vector(&self) -> &Vector<f64> {
        &self.pca_vector
    }

    pub fn pca_value(&self) -> &Vector<f64> {
        &self.pca_value
    }
}

/// Calculates PCA for the given matrix, wrapping OpenCV's PCA compute
///
/// # Arguments
/// * `points` - Points from image for PCA analysis
/// * `max_components` - Maximum number of PCA regions to return, 0 for unbounded
pub fn binary_pca(points: &[Point_<f64>], max_components: i32) -> Result<PcaData> {
    let mut image_data =
        Mat::new_rows_cols_with_default(points.len() as i32, 2, CV_64F, Scalar::default())?;
    (0..image_data.rows()).for_each(|idx| {
        *image_data.at_mut(idx * image_data.cols()).unwrap() = points[idx as usize].x;
        *image_data.at_mut(idx * image_data.cols() + 1).unwrap() = points[idx as usize].y;
    });

    let (mut mean, mut pca_vector, mut pca_value) =
        (Mat::default(), Mat::default(), Mat::default());
    pca_compute2(
        &image_data,
        &mut mean,
        &mut pca_vector,
        &mut pca_value,
        max_components, // C++ default parameter is 0
    )
    .unwrap();

    Ok(PcaData {
        mean: mean
            .iter()
            .unwrap()
            .map(|(_, x)| x)
            .collect::<Vector<f64>>(),
        pca_vector: pca_vector
            .iter()
            .unwrap()
            .map(|(_, x)| x)
            .collect::<Vector<f64>>(),
        pca_value: pca_value
            .iter()
            .unwrap()
            .map(|(_, x)| x)
            .collect::<Vector<f64>>(),
    })
}

/// Produces a new Mat with only full-value (255) points
///
/// # Arguments:
/// * `binary_image` - Grayscale Mat with only values of 255 and ! 255
///
/// # Examples
/// ```
/// use opencv::core::{Mat, Mat_, Point, VecN};
/// use opencv::prelude::MatTraitConstManual;
/// use sw8s_rust_lib::vision::image_prep::cvt_binary_to_points;
///
/// let raw_mat: [u8; 7] = [0, 0, 0, 255, 255, 0, 0];
/// let mat: Mat_<u8> = Mat::from_slice(&raw_mat).unwrap().clone_pointee().try_into_typed().unwrap();
///
/// let points: [Point; 2] = [
///     Point::from_vec2(VecN::from_array([0, 3])),
///     Point::from_vec2(VecN::from_array([0, 4])),
/// ];
///
/// /*
/// assert_eq!(
///     cvt_binary_to_points(&mat)
///         .unwrap()
///         .try_into_typed::<Point>()
///         .unwrap()
///         .data_typed()
///         .unwrap(),
///     points
/// );
/// */
/// ```
pub fn cvt_binary_to_points(binary_image: &Mat_<u8>) -> Vec<Point_<f64>> {
    (0..binary_image.rows())
        .flat_map(|row_idx| {
            (0..binary_image.cols()).filter_map(move |col_idx| {
                if *binary_image.at_2d(row_idx, col_idx).unwrap() == 255 {
                    Some(Point_::new(row_idx as f64, col_idx as f64))
                } else {
                    None
                }
            })
        })
        .collect()
}

/// Returns only unique colors in the Mat
///
/// # Arguments
/// * `image` - Mat to find unique colors in
///
/// # Examples:
/// ```
/// use opencv::core::{Mat, Mat_};
/// use opencv::prelude::MatTraitConstManual;
/// use sw8s_rust_lib::vision::image_prep::unique_colors;
///
/// let raw_mat: [u8; 7] = [0, 5, 32, 32, 5, 0, 1];
/// let mat: Mat_<u8> = Mat::from_slice(&raw_mat).unwrap().clone_pointee().try_into_typed().unwrap();
/// let unique = vec![0, 5, 32, 1];
///
/// assert_eq!(unique_colors(mat).unwrap(), unique);
/// ```
pub fn unique_colors<T>(image: Mat_<T>) -> Result<Vec<T>>
where
    T: DataType + Hash + Clone + Eq,
{
    Ok(image.data_typed()?.iter().unique().cloned().collect())
}

/// Wrapper for VecN that supports hashing
#[derive(Debug, PartialEq, Eq, Clone)]
struct VecNHash<T: Hash, const N: usize> {
    inner: VecN<T, N>,
}

impl<T: Hash, const N: usize> Deref for VecNHash<T, N> {
    type Target = VecN<T, N>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Hash, const N: usize> Hash for VecNHash<T, N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.iter().for_each(|x| x.hash(state));
    }
}

impl<T: Hash, const N: usize> VecNHash<T, N> {
    fn new(inner: VecN<T, N>) -> Self {
        VecNHash { inner }
    }
}

/// Returns only unique colors for Mat with > 1 dimension
///
/// # Arguments
/// * `image` - Mat of VecN to find unique colors in
///
/// # Examples:
/// ```
/// use opencv::core::{Mat, Mat_, VecN};
/// use opencv::prelude::MatTraitConstManual;
/// use sw8s_rust_lib::vision::image_prep::unique_colors_vec;
///
/// let raw_mat: [VecN<u8, 2>; 1] = [VecN::from_array([0, 0])];
/// let mat: Mat_<VecN<u8, 2>> = Mat::from_slice(&raw_mat).unwrap().clone_pointee().try_into_typed().unwrap();
/// let unique = vec![VecN::from_array([0, 0])];
///
/// assert_eq!(unique_colors_vec(mat).unwrap(), unique);
/// ```
pub fn unique_colors_vec<T, const N: usize>(image: Mat_<VecN<T, N>>) -> Result<Vec<VecN<T, N>>>
where
    T: DataType + Hash + Clone + Eq,
{
    Ok(image
        .data_typed()?
        .iter()
        .cloned()
        .map(VecNHash::new)
        .unique()
        .map(|x| *x)
        .collect())
}

pub fn kmeans(img: &Mat, n_clusters: i32, attempts: i32) -> Mat {
    let data = img.reshape(1, img.total() as i32).unwrap();
    let mut data_32f = Mat::default();
    data.convert_to(&mut data_32f, CV_32F, 1.0, 0.0).unwrap();
    let mut best_labels = Mat::default();
    let criteria = TermCriteria::new(1, 100, 1.0).unwrap(); // 1 -> COUNT
    let flags = KMEANS_PP_CENTERS;
    let mut center = Mat::default();

    opencv::core::kmeans(
        &data_32f,
        n_clusters,
        &mut best_labels,
        criteria,
        attempts,
        flags,
        &mut center,
    )
    .unwrap();
    let mut draw =
        Mat::new_size_with_default(Size::new(img.total() as i32, 1), CV_32FC3, VecN::default())
            .unwrap();
    let colors = center.reshape(3, n_clusters).unwrap();

    (0..n_clusters).for_each(|idx| {
        let mut mask = Mat::default();
        opencv::core::compare(&best_labels, &Scalar::from(idx), &mut mask, CMP_EQ).unwrap();
        let col = colors.row(idx).unwrap();
        // imgrot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        let mut mask_rot = Mat::default();
        opencv::core::rotate(&mask, &mut mask_rot, ROTATE_90_COUNTERCLOCKWISE).unwrap();
        let draw_vals = col.at::<VecN<f32, 3>>(0).unwrap();
        draw.set_to(draw_vals, &mask_rot).unwrap();
    });

    let draw = draw.reshape(3, img.rows()).unwrap();
    let mut draw_8u = Mat::default();
    draw.convert_to(&mut draw_8u, CV_8U, 1.0, 0.0).unwrap();
    draw_8u
}
