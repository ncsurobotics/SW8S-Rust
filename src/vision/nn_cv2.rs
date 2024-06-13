use anyhow::Result;
use assert_approx_eq::assert_approx_eq;
use derive_getters::Getters;
use opencv::{
    core::{
        min_max_loc, no_array, GpuMat, Point, Rect2d, Scalar, Size, VecN, Vector, CV_32F, CV_8S,
    },
    dnn::{blob_from_image, read_net_from_onnx, read_net_from_onnx_buffer, Net},
    prelude::{Mat, MatTraitConst, NetTrait, NetTraitConst},
};
use std::{fmt::Debug, sync::Mutex};
use std::{hash::Hash, ptr::null_mut};

#[cfg(feature = "cuda")]
use opencv::dnn::{DNN_BACKEND_CUDA, DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16};

#[cfg(feature = "cuda_min_max_loc")]
use opencv::cudaarithm::min_max_loc as cuda_min_max_loc;

#[derive(Debug, Clone, Getters, PartialEq)]
pub struct YoloDetection {
    class_id: i32,
    confidence: f64,
    bounding_box: Rect2d,
}

#[derive(Debug, Clone, Getters)]
pub struct YoloClass<T> {
    pub identifier: T,
    pub confidence: f64,
}

impl<T: PartialEq> PartialEq<T> for YoloClass<T> {
    fn eq(&self, other: &T) -> bool {
        self.identifier == *other
    }
}

impl<T: PartialEq> PartialEq for YoloClass<T> {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
    }
}

impl<T: PartialEq> Eq for YoloClass<T> {}

impl<T: Hash> Hash for YoloClass<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identifier.hash(state)
    }
}

impl<T> TryFrom<YoloDetection> for YoloClass<T>
where
    T: TryFrom<i32>,
    <T as TryFrom<i32>>::Error: std::error::Error + Send + Sync,
{
    type Error = T::Error;
    fn try_from(val: YoloDetection) -> Result<Self, Self::Error> {
        Ok(YoloClass {
            identifier: val.class_id.try_into()?,
            confidence: val.confidence,
        })
    }
}

pub trait VisionModel: Debug + Sync + Send {
    fn detect_yolo_v5(&mut self, image: &Mat, threshold: f64) -> Vec<YoloDetection>;
    fn size(&self) -> Size;
}

/* -------------------------------------------------- */
/* --------------- ONNX implementation -------------- */
/* -------------------------------------------------- */

/// ONNX vision model running via OpenCV
#[derive(Debug)]
pub struct OnnxModel {
    net: Mutex<Net>,
    //out_blob_names: Vec<String>,
    num_objects: usize,
    //output: Vec<usize>,
    //output_description: Vec<Rect2d>,
    model_size: Size,
    factor: f64,
}

impl OnnxModel {
    /// Creates model from in memory byte buffer
    ///
    /// # Arguments:
    /// * `model_bytes` - ONNX model in u8
    /// * `model_size` - input image square dimensions (e.g. 640 for 640x640)
    /// * `num_objects` - number of objects model can output
    ///
    /// # Examples:
    ///
    /// ```
    /// use opencv::core::Vector;
    /// use sw8s_rust_lib::vision::nn_cv2::OnnxModel;
    ///
    /// OnnxModel::from_bytes(
    ///     &Vector::from_slice(include_bytes!("models/buoy_320.onnx")),
    ///     320,
    ///     4,
    /// )
    /// .unwrap();
    /// ```
    pub fn from_bytes(
        model_bytes: &Vector<u8>,
        model_size: i32,
        num_objects: usize,
    ) -> Result<Self> {
        let mut net = read_net_from_onnx_buffer(model_bytes)?;
        #[cfg(feature = "cuda")]
        {
            net.set_preferable_backend(DNN_BACKEND_CUDA)?;
            if cfg!(feature = "cuda_f16") {
                net.set_preferable_target(DNN_TARGET_CUDA_FP16)?;
            } else {
                net.set_preferable_target(DNN_TARGET_CUDA)?;
            }
        }

        Ok(Self {
            net: Mutex::new(net),
            num_objects,
            model_size: Size::new(model_size, model_size),
            factor: Self::size_to_factor(model_size),
        })
    }

    /// Creates model from file (use a local path)
    ///
    /// # Arguments:
    /// * `model_name` - path to ONNX model (from working directory)
    /// * `model_size` - input image square dimensions (e.g. 640 for 640x640)
    /// * `num_objects` - number of objects model can output
    ///
    /// # Examples:
    /// ```
    /// use opencv::core::Vector;
    /// use sw8s_rust_lib::vision::nn_cv2::OnnxModel;
    ///
    /// OnnxModel::from_file("src/vision/models/buoy_320.onnx", 320, 4).unwrap();
    /// ```
    pub fn from_file(model_name: &str, model_size: i32, num_objects: usize) -> Result<Self> {
        let mut net = read_net_from_onnx(model_name)?;
        #[cfg(feature = "cuda")]
        {
            net.set_preferable_backend(DNN_BACKEND_CUDA)?;
            if cfg!(feature = "cuda_f16") {
                net.set_preferable_target(DNN_TARGET_CUDA_FP16)?;
            } else {
                net.set_preferable_target(DNN_TARGET_CUDA)?;
            }
        }

        Ok(Self {
            net: Mutex::new(net),
            num_objects,
            model_size: Size::new(model_size, model_size),
            factor: Self::size_to_factor(model_size),
        })
    }

    /// Calculates coordinate factor based on model size
    fn size_to_factor(model_size: i32) -> f64 {
        640.0 / model_size as f64
    }

    fn get_output_names(net: &Net) -> Vector<String> {
        let out_layers = net
            .get_unconnected_out_layers()
            .expect("Error getting unconnected out layers from model.");
        let layer_names = net
            .get_layer_names()
            .expect("Error getting layer names from model.");

        Vector::from_iter(
            out_layers
                .iter()
                .map(|layer_num| layer_names.get((layer_num - 1) as usize).unwrap())
                .to_owned()
                .collect::<Vec<_>>()
                .iter()
                .map(|s| s.as_str()),
        )
    }

    pub fn get_net(&mut self) -> &mut Net {
        self.net.get_mut().unwrap()
    }

    pub fn get_model_size(&self) -> Size {
        self.model_size
    }
}

/// Loads model from file, mostly at compile time
///
/// # Arguments:
/// * `model_name` - path to ONNX model (relative to file function is called from)
/// * `model_size` - input image square dimensions (e.g. 640 for 640x640)
/// * `num_objects` - number of objects model can output
///
/// # Examples:
///
/// ```
/// use sw8s_rust_lib::{
///     load_onnx,
///     vision::nn_cv2::OnnxModel,
/// };
///
/// let model: OnnxModel = load_onnx!("models/buoy_320.onnx", 320, 4);
/// ```
#[macro_export]
macro_rules! load_onnx {
    ($model_name:expr, $model_size:expr, $num_objects:expr) => {{
        use opencv::core::Vector;

        OnnxModel::from_bytes(
            &Vector::from_slice(include_bytes!($model_name)),
            $model_size,
            $num_objects,
        )
        .unwrap()
    }};
}

impl VisionModel for OnnxModel {
    fn detect_yolo_v5(&mut self, image: &Mat, threshold: f64) -> Vec<YoloDetection> {
        let mut result: Vector<Mat> = Vector::new();
        let result_names = Self::get_output_names(&self.net.lock().unwrap());
        let blob = blob_from_image(
            image,
            1.0 / 255.0,
            self.model_size,
            Scalar::from(0.0),
            true,
            false,
            CV_32F,
        )
        .unwrap();

        self.net
            .lock()
            .unwrap()
            .set_input(&blob, "", 1.0, Scalar::from(0.0))
            .unwrap();
        self.net
            .lock()
            .unwrap()
            .forward(&mut result, &result_names)
            .unwrap();

        let cuda_result = self.process_net_cuda(&result, threshold as f32);
        let actual_result = self.process_net(result, threshold);
        //println!("Actual Result: {:#?}", actual_result);
        /*
        println!(
            "CUDA Len: {}, Actual Len: {}",
            cuda_result.len(),
            actual_result.len()
        );
        */
        for i in 0..actual_result.len() {
            //println!("Testing {i}");
            let cuda = &cuda_result[i];
            let actual = &actual_result[i];
            assert_eq!(cuda.class_id, actual.class_id);
            assert_approx_eq!(cuda.confidence, actual.confidence, 1e-2);
            assert_approx_eq!(cuda.bounding_box.x, actual.bounding_box.x, 1e-2);
            assert_approx_eq!(cuda.bounding_box.y, actual.bounding_box.y, 1e-2);
            assert_approx_eq!(cuda.bounding_box.width, actual.bounding_box.width, 1e-2);
            assert_approx_eq!(cuda.bounding_box.height, actual.bounding_box.height, 1e-2);
            assert_eq!(cuda.class_id, actual.class_id);
            //assert_eq!(cuda_result[i], actual_result[i]);
        }
        actual_result
    }

    fn size(&self) -> Size {
        self.model_size
    }
}

impl OnnxModel {
    /// Returns all detections from a net's output
    ///
    /// # Arguments
    /// * `result` - iterator of net output
    /// * `threshold` - minimum confidence
    fn process_net<I>(&self, result: I, threshold: f64) -> Vec<YoloDetection>
    where
        I: IntoIterator<Item = Mat>,
    {
        result
            .into_iter()
            .flat_map(|level| -> Vec<YoloDetection> {
                // This reshape is always valid as per the model design
                let level = level
                    .reshape(1, (level.total() / (5 + self.num_objects)) as i32)
                    .unwrap();

                (0..level.rows())
                    .map(|idx| level.row(idx).unwrap())
                    .filter_map(|row| -> Option<YoloDetection> {
                        // Cols is always > 5. The column range can always be constructed, since
                        // row always has level.cols number of columns.
                        let scores = row
                            .col_range(&opencv::core::Range::new(5, level.cols()).unwrap())
                            .unwrap();

                        let mut max_loc = Point::default();

                        // There is always a global minimum and maximum
                        if cfg!(feature = "cuda_min_max_loc") {
                            // C++ interface wants null pointers for not returning a value...
                            #[allow(deref_nullptr)]
                            #[cfg(feature = "cuda_min_max_loc")]
                            unsafe {
                                cuda_min_max_loc(
                                    &scores,
                                    &mut *null_mut(),
                                    &mut *null_mut(),
                                    &mut *null_mut(),
                                    &mut max_loc,
                                    &no_array(),
                                )
                                .unwrap();
                            }
                        } else {
                            min_max_loc(&scores, None, None, None, Some(&mut max_loc), &no_array())
                                .unwrap();
                        }

                        // Always a valid index access
                        let confidence: f64 = row.at::<VecN<f32, 1>>(4).unwrap()[0].into();

                        if confidence > threshold {
                            // The given constant values are always valid indicies
                            let adjust_base = |idx: i32| -> f64 {
                                f64::from(row.at::<VecN<f32, 1>>(idx).unwrap()[0]) * self.factor
                            };

                            let x_adjust = |idx: i32| -> f64 { adjust_base(idx) / 640.0 * 800.0 };
                            let y_adjust = |idx: i32| -> f64 { adjust_base(idx) / 640.0 * 600.0 };

                            let (center_x, center_y, width, height) =
                                (x_adjust(0), y_adjust(1), x_adjust(2), y_adjust(3));

                            let left = center_x - width / 2.0;
                            let top = center_y - height / 2.0;

                            Some(YoloDetection {
                                class_id: max_loc.x,
                                confidence,
                                bounding_box: Rect2d {
                                    x: left,
                                    y: top,
                                    width,
                                    height,
                                },
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn process_net_cuda(&self, result: &Vector<Mat>, threshold: f32) -> Vec<YoloDetection> {
        #[derive(Debug)]
        #[repr(C)]
        struct CudaFormatMat {
            rows: i32,
            cols: i32,
            bytes: *const u8,
        }

        #[derive(Debug)]
        #[repr(C)]
        pub struct YoloDetectionCuda {
            confidence: f64,
            x: f64,
            y: f64,
            width: f64,
            height: f64,
            class_id: i32,
        }

        let mut total_rows = 0;

        let result = result
            .iter()
            .map(|level| -> CudaFormatMat {
                // This reshape is always valid as per the model design
                let level = level
                    .reshape(1, (level.total() / (5 + self.num_objects)) as i32)
                    .unwrap();

                total_rows += level.rows() as usize;

                CudaFormatMat {
                    bytes: level.data(),
                    rows: level.rows(),
                    cols: level.cols(),
                }
            })
            .collect::<Vec<_>>();

        let mut processed_detects = Vec::with_capacity(total_rows);
        let mut processed_valid = Vec::with_capacity(total_rows);
        unsafe {
            processed_detects.set_len(total_rows);
            processed_valid.set_len(total_rows);
        }

        #[link(name = "sw8s_cuda", kind = "static")]
        extern "C" {
            fn process_net_kernel(
                result: *const CudaFormatMat,
                num_levels: usize,
                threshold: f32,
                factor: f32,
                total_rows: usize,
                processed_detects: *mut YoloDetectionCuda,
                processed_valid: *mut bool,
            );
        }
        unsafe {
            process_net_kernel(
                result.as_ptr(),
                result.len(),
                threshold,
                self.factor as f32,
                total_rows,
                processed_detects.as_mut_ptr(),
                processed_valid.as_mut_ptr(),
            );
        }

        let full_processed = processed_valid
            .iter()
            .zip(processed_detects)
            .filter(|(status, _)| **status)
            .map(|(_, cuda_format)| YoloDetection {
                class_id: cuda_format.class_id,
                confidence: cuda_format.confidence,
                bounding_box: Rect2d {
                    x: cuda_format.x,
                    y: cuda_format.y,
                    width: cuda_format.width,
                    height: cuda_format.height,
                },
            })
            .collect();

        //println!("Fully processed: {:#?}", full_processed);

        full_processed
    }
}
