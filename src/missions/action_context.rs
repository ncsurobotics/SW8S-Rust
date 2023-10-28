use async_trait::async_trait;
use core::fmt::Debug;
use opencv::prelude::Mat;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio_serial::SerialStream;

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
pub struct EmptyActionContext;

pub struct FullActionContext<'a, T: AsyncWriteExt + Unpin + Send, U: MatSource> {
    control_board: &'a ControlBoard<T>,
    main_electronics_board: &'a MainElectronicsBoard,
    frame_source: &'a U,
}

impl<'a, T: AsyncWriteExt + Unpin + Send, U: MatSource> FullActionContext<'a, T, U> {
    pub const fn new(
        control_board: &'a ControlBoard<T>,
        main_electronics_board: &'a MainElectronicsBoard,
        frame_source: &'a U,
    ) -> Self {
        Self {
            control_board,
            main_electronics_board,
            frame_source,
        }
    }
}

impl<T: MatSource> GetControlBoard<WriteHalf<SerialStream>>
    for FullActionContext<'_, WriteHalf<SerialStream>, T>
{
    fn get_control_board(&self) -> &ControlBoard<WriteHalf<SerialStream>> {
        self.control_board
    }
}

impl<T: MatSource> GetMainElectronicsBoard for FullActionContext<'_, WriteHalf<SerialStream>, T> {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard {
        self.main_electronics_board
    }
}

#[async_trait]
impl<T: AsyncWriteExt + Unpin + Send, U: MatSource> MatSource for FullActionContext<'_, T, U> {
    async fn get_mat(&self) -> Mat {
        self.frame_source.get_mat().await
    }
}
