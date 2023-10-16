use core::fmt::Debug;
use tokio::io::AsyncWriteExt;

use crate::comms::{control_board::ControlBoard, meb::MainElectronicsBoard};

/**
 * Inherit this trait if you have a control board
 */
pub trait GetControlBoard<T: AsyncWriteExt + Unpin> {
    fn get_control_board(&self) -> &ControlBoard<T>;
}

/**
 * Inherit this trait if you have a MEB
 */
pub trait GetMainElectronicsBoard {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard;
}

#[derive(Debug)]
struct EmptyActionContext;

struct FullActionContext<T: AsyncWriteExt + Unpin + Send> {
    control_board: ControlBoard<T>,
    main_electronics_board: MainElectronicsBoard,
}

impl<T: AsyncWriteExt + Unpin + Send> FullActionContext<T> {
    const fn new(
        control_board: ControlBoard<T>,
        main_electronics_board: MainElectronicsBoard,
    ) -> Self {
        Self {
            control_board,
            main_electronics_board,
        }
    }
}

impl GetControlBoard<tokio_serial::SerialStream> for FullActionContext<tokio_serial::SerialStream> {
    fn get_control_board(&self) -> &ControlBoard<tokio_serial::SerialStream> {
        &self.control_board
    }
}

impl GetMainElectronicsBoard for FullActionContext<tokio_serial::SerialStream> {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard {
        &self.main_electronics_board
    }
}
