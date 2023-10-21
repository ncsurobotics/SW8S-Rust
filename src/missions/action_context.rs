use async_trait::async_trait;
use core::fmt::Debug;
use opencv::prelude::Mat;
use tokio::io::AsyncWriteExt;

use crate::{
    comms::{control_board::ControlBoard, meb::MainElectronicsBoard},
    video_source::MatSource,
};

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
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard;
}

#[derive(Debug)]
struct EmptyActionContext;

struct FullActionContext<T: AsyncWriteExt + Unpin + Send, U: MatSource> {
    control_board: ControlBoard<T>,
    main_electronics_board: MainElectronicsBoard,
    frame_source: U,
}

impl<T: AsyncWriteExt + Unpin + Send, U: MatSource> FullActionContext<T, U> {
    const fn new(
        control_board: ControlBoard<T>,
        main_electronics_board: MainElectronicsBoard,
        frame_source: U,
    ) -> Self {
        Self {
            control_board,
            main_electronics_board,
            frame_source,
        }
    }
}

impl<T: MatSource> GetControlBoard<tokio_serial::SerialStream>
    for FullActionContext<tokio_serial::SerialStream, T>
{
    fn get_control_board(&self) -> &ControlBoard<tokio_serial::SerialStream> {
        &self.control_board
    }
}

impl<T: MatSource> GetMainElectronicsBoard for FullActionContext<tokio_serial::SerialStream, T> {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard {
        &self.main_electronics_board
    }
}

#[async_trait]
impl<T: AsyncWriteExt + Unpin + Send, U: MatSource> MatSource for FullActionContext<T, U> {
    async fn get_mat(&self) -> Mat {
        self.frame_source.get_mat().await
    }
}
