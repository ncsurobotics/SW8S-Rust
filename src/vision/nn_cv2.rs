use anyhow::Result;
use derive_getters::Getters;
use opencv::{
    core::{Rect2d, Scalar, Size, VecN, Vector, CV_32F},
    dnn::{blob_from_image, read_net_from_onnx, read_net_from_onnx_buffer, Net},
    prelude::{Mat, MatTraitConst, NetTrait, NetTraitConst},
};
use std::hash::Hash;
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::Mutex,
};

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

pub trait VisionModel: Debug + Sync + Send + Clone {
    type PostProcessArgs;
    type ModelOutput;

    /// Forward pass the matrix through the model, skipping post-processing
    fn forward(&mut self, image: &Mat) -> Self::ModelOutput;
    /// Convert output from a model into detections
    fn post_process_args(&self) -> Self::PostProcessArgs;
    fn post_process(
        args: Self::PostProcessArgs,
        output: Self::ModelOutput,
        threshold: f64,
    ) -> Vec<YoloDetection>;

    /// Full input -> output processing
    fn detect_yolo_v5(&mut self, image: &Mat, threshold: f64) -> Vec<YoloDetection> {
        let model_output = self.forward(image);
        Self::post_process(self.post_process_args(), model_output, threshold)
    }
    fn size(&self) -> Size;
}

/* -------------------------------------------------- */
/* --------------- ONNX implementation -------------- */
/* -------------------------------------------------- */

/// Wrapper to let Rust know Net is Send and Sync.
///
/// We know it doesn't rely on thread-local state, but Rust assumes all
/// pointers are not Send or Sync by default.
#[derive(Debug, Clone)]
struct NetWrapper(pub Net);

impl Deref for NetWrapper {
    type Target = Net;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for NetWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl Send for NetWrapper {}
unsafe impl Sync for NetWrapper {}

/// ONNX vision model running via OpenCV
#[derive(Debug)]
pub struct OnnxModel {
    net: Mutex<NetWrapper>,
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
        let net = read_net_from_onnx_buffer(model_bytes)?;
        /*
        #[cfg(feature = "cuda")]
        {
            net.set_preferable_backend(DNN_BACKEND_CUDA)?;
            if cfg!(feature = "cuda_f16") {
                net.set_preferable_target(DNN_TARGET_CUDA_FP16)?;
            } else {
                net.set_preferable_target(DNN_TARGET_CUDA)?;
            }
        }
        */

        Ok(Self {
            net: Mutex::new(NetWrapper(net)),
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
        let net = read_net_from_onnx(model_name)?;
        /*
        #[cfg(feature = "cuda")]
        {
            net.set_preferable_backend(DNN_BACKEND_CUDA)?;
            if cfg!(feature = "cuda_f16") {
                net.set_preferable_target(DNN_TARGET_CUDA_FP16)?;
            } else {
                net.set_preferable_target(DNN_TARGET_CUDA)?;
            }
        }
        */

        Ok(Self {
            net: Mutex::new(NetWrapper(net)),
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

impl Clone for OnnxModel {
    fn clone(&self) -> Self {
        Self {
            net: Mutex::new(self.net.lock().unwrap().clone()),
            num_objects: self.num_objects,
            model_size: self.model_size,
            factor: self.factor,
        }
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
        let result = self.forward(image);

        #[cfg(feature = "cuda")]
        let post_processing = Self::process_net_cuda(
            self.num_objects,
            self.factor as f32,
            &result,
            threshold as f32,
        );

        #[cfg(not(feature = "cuda"))]
        let post_processing = Self::process_net(self.num_objects, self.factor, result, threshold);

        post_processing
    }

    fn forward(&mut self, image: &Mat) -> Self::ModelOutput {
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

        result
    }

    type ModelOutput = Vector<Mat>;

    #[cfg(feature = "cuda")]
    type PostProcessArgs = (usize, f32);
    #[cfg(not(feature = "cuda"))]
    type PostProcessArgs = (usize, f64);

    fn post_process_args(&self) -> Self::PostProcessArgs {
        #[cfg(feature = "cuda")]
        {
            (self.num_objects, self.factor as f32)
        }
        #[cfg(not(feature = "cuda"))]
        {
            (self.num_objects, self.factor)
        }
    }

    fn post_process(
        args: Self::PostProcessArgs,
        output: Self::ModelOutput,
        threshold: f64,
    ) -> Vec<YoloDetection> {
        #[cfg(feature = "cuda")]
        let post_processing = Self::process_net_cuda(args.0, args.1, &output, threshold as f32);

        #[cfg(not(feature = "cuda"))]
        let post_processing = Self::process_net(args.0, args.1, output, threshold);

        post_processing
    }

    fn size(&self) -> Size {
        self.model_size
    }
}

impl OnnxModel {
    #[allow(unused)]
    /// Returns all detections from a net's output
    ///
    /// # Arguments
    /// * `result` - iterator of net output
    /// * `threshold` - minimum confidence
    fn process_net<I>(
        num_objects: usize,
        factor: f64,
        result: I,
        threshold: f64,
    ) -> Vec<YoloDetection>
    where
        I: IntoIterator<Item = Mat>,
    {
        result
            .into_iter()
            .flat_map(|level| -> Vec<YoloDetection> {
                // This reshape is always valid as per the model design
                let level = level
                    .reshape(1, (level.total() / (5 + num_objects)) as i32)
                    .unwrap();

                (0..level.rows())
                    .map(|idx| level.row(idx).unwrap())
                    .filter_map(|row| -> Option<YoloDetection> {
                        // Cols is always > 5. The column range can always be constructed, since
                        // row always has level.cols number of columns.
                        let scores = row
                            .col_range(&opencv::core::Range::new(5, level.cols()).unwrap())
                            .unwrap();

                        let mut max_loc = 5;
                        for idx in 6..level.cols() {
                            if row.at::<VecN<f32, 1>>(max_loc).unwrap()[0]
                                < row.at::<VecN<f32, 1>>(idx).unwrap()[0]
                            {
                                max_loc = idx;
                            }
                        }
                        max_loc -= 5;

                        // Always a valid index access
                        let confidence: f64 = row.at::<VecN<f32, 1>>(4).unwrap()[0].into();

                        if confidence > threshold {
                            // The given constant values are always valid indicies
                            let adjust_base = |idx: i32| -> f64 {
                                f64::from(row.at::<VecN<f32, 1>>(idx).unwrap()[0]) * factor
                            };

                            let x_adjust = |idx: i32| -> f64 { adjust_base(idx) / 640.0 * 800.0 };
                            let y_adjust = |idx: i32| -> f64 { adjust_base(idx) / 640.0 * 600.0 };

                            let (center_x, center_y, width, height) =
                                (x_adjust(0), y_adjust(1), x_adjust(2), y_adjust(3));

                            let left = center_x - width / 2.0;
                            let top = center_y - height / 2.0;

                            Some(YoloDetection {
                                class_id: max_loc,
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

    /// Alternative to [`process_net`] that uses a CUDA kernel
    #[cfg(feature = "cuda")]
    fn process_net_cuda(
        num_objects: usize,
        factor: f32,
        result: &Vector<Mat>,
        threshold: f32,
    ) -> Vec<YoloDetection> {
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
                    .reshape(1, (level.total() / (5 + num_objects)) as i32)
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
                factor,
                total_rows,
                processed_detects.as_mut_ptr(),
                processed_valid.as_mut_ptr(),
            );
        }

        processed_valid
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
            .collect()
    }
}

/*
/// Utility struct for [`ModelPipelined`].
///
/// * `mat`: latest available matrix. Set to default on read.
/// * `dropped`: tracks if ModelPipelined is dropped, for thread cleanup.
#[derive(Debug)]
struct ModelPipelinedInput {
    pub mat: Box<[u8]>,
    pub dropped: bool,
}

/// [`OnnxModel`] that pipelines processing in blocking threads.
///
/// The input is processed on blocking threads, and only the newest available
/// input should be processed, so `input_mut` is used for threads to claim
/// whenever an unclaimed new input is available. It also tracks for when to
/// drop the threads.
///
/// The output is asynchronous, written to with blocking synchronous calls from
/// the post processing stage.
#[derive(Debug)]
pub struct ModelPipelined {
    input_mut: Arc<(Condvar, Mutex<ModelPipelinedInput>)>,
    output_ch: async_channel::Receiver<Vec<YoloDetection>>,
}

impl ModelPipelined {
    /// Pipelines model processing in blocking threads.
    ///
    /// # Parameters
    /// * `model`: A model to be cloned into threads.
    /// * `model_threads`: Number of threads with processing models.
    /// * `post_processing_threads`: Number of threads converting model output.
    /// * `threshold`: [0, 1] minimum score for a detection.
    pub async fn new<T>(
        model: T,
        model_threads: NonZeroUsize,
        post_processing_threads: NonZeroUsize,
        threshold: f64,
    ) -> Self
    where
        T: VisionModel<ModelOutput = Vector<Mat>>
            + Clone
            + Send
            + Sync
            + 'static
            + opencv::prelude::DataType,
        T::PostProcessArgs: Send + Clone,
    {
        let input_mut = Arc::new((
            Condvar::new(),
            Mutex::new(ModelPipelinedInput {
                mat: Box::new([]),
                dropped: false,
            }),
        ));
        let (output_tx, output_ch) = async_channel::unbounded();

        // Both processing threads are blocking, so using a sync structure.
        let (inner_tx, inner_rx) = crossbeam::channel::unbounded();

        for _ in 0..model_threads.into() {
            let mut model = model.clone();
            let input_mut = input_mut.clone();
            let inner_tx: crossbeam::channel::Sender<Box<[Box<[T]>]>> = inner_tx.clone();

            spawn_blocking(move || loop {
                let input = Mat::from_slice(&{
                    // When we get a notification on this thread, new data can
                    // always be directly claimed.
                    let mut guard = input_mut.1.lock().unwrap();
                    guard = input_mut.0.wait(guard).unwrap();

                    // Exit this thread if the struct was dropped
                    if guard.dropped {
                        break;
                    };

                    // Move the matrix to local memory to avoid holding up the
                    // lock. The default value should never be read by another
                    // thread.
                    std::mem::take(&mut guard.mat)
                })
                .unwrap()
                .clone_pointee();

                if !input.is_allocated() {
                    continue;
                }

                // Hand off to post processing
                let forwarded = model.forward(&input);
                let boxed = forwarded
                    .into_iter()
                    .map(|x| {
                        x.to_vec_2d()
                            .unwrap()
                            .into_iter()
                            .flatten()
                            .collect_vec()
                            .into_boxed_slice()
                    })
                    .collect_vec()
                    .into_boxed_slice();
                if inner_tx.send(boxed).is_err() {
                    break;
                };
            });
        }

        for _ in 0..post_processing_threads.into() {
            let inner_rx = inner_rx.clone();
            let output_tx = output_tx.clone();
            let post_process_args = model.post_process_args();

            spawn_blocking(move || {
                // Thread exits when model output threads exit (struct drop).
                while let Ok(input) = inner_rx.recv() {
                    let input = input
                        .into_iter()
                        .map(|x| Mat::from_slice(&x).unwrap().clone_pointee())
                        .collect();
                    let post_process_args = post_process_args.clone();
                    let processed_output =
                        T::post_process(post_process_args.clone(), input, threshold);
                    // Blocking call on this end, async on the other.
                    // Never stalls for capacity, since output is unbounded.
                    if output_tx.send_blocking(processed_output).is_err() {
                        break;
                    };
                }
            });
        }

        Self {
            input_mut,
            output_ch,
        }
    }

    /// Update the model with a newer [`Mat`] to process.
    pub fn update_mat(&self, mat: Mat) -> &Self {
        let mut input = self.input_mut.1.lock().unwrap();
        input.mat = mat
            .to_vec_2d()
            .unwrap()
            .into_iter()
            .flatten()
            .collect_vec()
            .into_boxed_slice();
        self.input_mut.0.notify_one();
        self
    }

    /// Get the oldest available output.
    ///
    /// Stalls until an output is available.
    pub async fn get_single(&self) -> Vec<YoloDetection> {
        self.output_ch.recv().await.unwrap()
    }

    /// Get the oldest N available outputs.
    ///
    /// Stalls until N outputs are available.
    /// Returns in order oldest -> newest.
    pub async fn get_multiple(&self, count: usize) -> Vec<Vec<YoloDetection>> {
        let mut output = Vec::with_capacity(count);
        for _ in 0..count {
            output.push(self.output_ch.recv().await.unwrap())
        }
        output
    }

    /// Get the newest N available outputs.
    ///
    /// Stalls until N outputs are available.
    /// Returns in order oldest -> newest.
    pub async fn get_multiple_newest(&self, count: usize) -> Vec<Vec<YoloDetection>> {
        let mut output = Vec::with_capacity(count);
        for _ in 0..count {
            output.push(self.output_ch.recv().await.unwrap())
        }
        output.extend(iter::from_fn(|| self.output_ch.try_recv().ok()));

        output.into_iter().rev().take(count).rev().collect()
    }

    /// Get all available output immediately.
    pub async fn get_all(&self) -> Vec<Vec<YoloDetection>> {
        iter::from_fn(|| self.output_ch.try_recv().ok()).collect()
    }
}

impl Drop for ModelPipelined {
    /// Trigger thread cleanup.
    fn drop(&mut self) {
        self.input_mut.1.lock().unwrap().dropped = true;
        self.input_mut.0.notify_all();
    }
}
*/
