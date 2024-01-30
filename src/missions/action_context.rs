use async_trait::async_trait;
use core::fmt::Debug;
use opencv::core::Mat;
use tokio::io::AsyncWriteExt;
use tokio::io::{AsyncWrite, WriteHalf};

use crate::video_source::appsink::Camera;
use crate::video_source::MatSource;
use crate::{
    comms::stubborn_serial::StubbornSerialStream,
    comms::{control_board::ControlBoard, meb::MainElectronicsBoard},
};

/**
 * Inherit this trait if you have a control board
 */
pub trait GetControlBoard<T: AsyncWrite + Unpin>: Send + Sync {
    fn get_control_board(&self) -> &ControlBoard<T>;
}

/**
 * Inherit this trait if you have a MEB
 */
pub trait GetMainElectronicsBoard: Send + Sync {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard;
}

/**
 * Inherit this trait if you have a front camera
 */
#[async_trait]
pub trait GetFrontCamMat {
    async fn get_front_camera_mat(&self) -> Mat;
}

/**
 * Inherit this trait if you have a bottom camera
 */
#[async_trait]
pub trait GetBottomCamMat {
    async fn get_bottom_camera_mat(&self) -> Mat;
}

#[derive(Debug)]
pub struct EmptyActionContext;

pub struct FullActionContext<'a, T: AsyncWriteExt + Unpin + Send> {
    control_board: &'a ControlBoard<T>,
    main_electronics_board: &'a MainElectronicsBoard,
    front_cam: &'a Camera,
    bottom_cam: &'a Camera,
}

impl<'a, T: AsyncWriteExt + Unpin + Send> FullActionContext<'a, T> {
    pub const fn new(
        control_board: &'a ControlBoard<T>,
        main_electronics_board: &'a MainElectronicsBoard,
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

impl GetControlBoard<WriteHalf<StubbornSerialStream>>
    for FullActionContext<'_, WriteHalf<StubbornSerialStream>>
{
    fn get_control_board(&self) -> &ControlBoard<WriteHalf<StubbornSerialStream>> {
        self.control_board
    }
}

impl GetMainElectronicsBoard for FullActionContext<'_, WriteHalf<StubbornSerialStream>> {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard {
        self.main_electronics_board
    }
}

#[async_trait]
impl<T: AsyncWriteExt + Unpin + Send> GetFrontCamMat for FullActionContext<'_, T> {
    async fn get_front_camera_mat(&self) -> Mat {
        self.front_cam.get_mat().await
    }
}

#[async_trait]
impl<T: AsyncWriteExt + Unpin + Send> GetBottomCamMat for FullActionContext<'_, T> {
    async fn get_bottom_camera_mat(&self) -> Mat {
        self.bottom_cam.get_mat().await
    }
}
