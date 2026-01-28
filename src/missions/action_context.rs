use core::fmt::Debug;
use opencv::core::Mat;
#[cfg(feature = "annotated_streams")]
use opencv::mod_prelude::ToInputArray;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio_serial::SerialStream;

use crate::comms::{control_board::ControlBoard, meb::MainElectronicsBoard};
use crate::video_source::appsink::Camera;
use crate::video_source::MatSource;
/**
 * Inherit this trait if you have a control board
 */
pub trait GetControlBoard<T: AsyncWriteExt + Unpin>: Send + Sync {
    fn get_control_board(&self) -> &ControlBoard<T>;
}

/**
 * Inherit this trait if you have a MEB
 */
pub trait GetMainElectronicsBoard: Send + Sync {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard<WriteHalf<SerialStream>>;
}

/**
 * Inherit this trait if you have a front camera
 */
#[allow(async_fn_in_trait)]
pub trait FrontCamIO {
    fn get_front_camera_mat(&self) -> impl std::future::Future<Output = Mat> + Send;
    #[cfg(feature = "annotated_streams")]
    async fn annotate_front_camera(&self, image: &impl ToInputArray);
}

/**
 * Inherit this trait if you have a bottom camera
 */
#[allow(async_fn_in_trait)]
pub trait BottomCamIO {
    async fn get_bottom_camera_mat(&self) -> Mat;
    #[cfg(feature = "annotated_streams")]
    async fn annotate_bottom_camera(&self, image: &impl ToInputArray);
}

#[derive(Debug)]
pub struct EmptyActionContext;

impl Unpin for EmptyActionContext {
    // add code here
}

pub struct FullActionContext<'a, T: AsyncWriteExt + Unpin + Send> {
    control_board: &'a ControlBoard<T>,
    main_electronics_board: &'a MainElectronicsBoard<WriteHalf<SerialStream>>,
    front_cam: &'a Camera,
    bottom_cam: &'a Camera,
}

impl<'a, T: AsyncWriteExt + Unpin + Send> FullActionContext<'a, T> {
    pub const fn new(
        control_board: &'a ControlBoard<T>,
        main_electronics_board: &'a MainElectronicsBoard<WriteHalf<SerialStream>>,
        front_cam: &'a Camera,
        bottom_cam: &'a Camera,
    ) -> Self {
        Self {
            control_board,
            main_electronics_board,
            front_cam,
            bottom_cam,
        }
    }
}

impl GetControlBoard<WriteHalf<SerialStream>> for FullActionContext<'_, WriteHalf<SerialStream>> {
    fn get_control_board(&self) -> &ControlBoard<WriteHalf<SerialStream>> {
        self.control_board
    }
}

impl GetMainElectronicsBoard for FullActionContext<'_, WriteHalf<SerialStream>> {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard<WriteHalf<SerialStream>> {
        self.main_electronics_board
    }
}

impl<T: AsyncWriteExt + Unpin + Send> FrontCamIO for FullActionContext<'_, T> {
    async fn get_front_camera_mat(&self) -> Mat {
        self.front_cam.get_mat().await
    }
    #[cfg(feature = "annotated_streams")]
    async fn annotate_front_camera(&self, image: &impl ToInputArray) {
        self.front_cam.push_annotated_frame(image);
    }
}

impl<T: AsyncWriteExt + Unpin + Send> BottomCamIO for FullActionContext<'_, T> {
    async fn get_bottom_camera_mat(&self) -> Mat {
        self.bottom_cam.get_mat().await
    }
    #[cfg(feature = "annotated_streams")]
    async fn annotate_bottom_camera(&self, image: &impl ToInputArray) {
        self.bottom_cam.push_annotated_frame(image);
    }
}

impl GetControlBoard<WriteHalf<SerialStream>> for EmptyActionContext {
    fn get_control_board(&self) -> &ControlBoard<WriteHalf<SerialStream>> {
        todo!()
    }
}

impl GetMainElectronicsBoard for EmptyActionContext {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard<WriteHalf<SerialStream>> {
        todo!()
    }
}

impl FrontCamIO for EmptyActionContext {
    async fn get_front_camera_mat(&self) -> Mat {
        todo!()
    }
    #[cfg(feature = "annotated_streams")]
    async fn annotate_front_camera(&self, _image: &impl ToInputArray) {
        todo!();
    }
}

impl BottomCamIO for EmptyActionContext {
    async fn get_bottom_camera_mat(&self) -> Mat {
        todo!()
    }
    #[cfg(feature = "annotated_streams")]
    async fn annotate_bottom_camera(&self, _image: &impl ToInputArray) {
        todo!();
    }
}
