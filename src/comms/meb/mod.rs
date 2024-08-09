use std::sync::Arc;

use anyhow::Result;
use futures::io::ReadHalf;
use tokio::{
    io::{AsyncReadExt, AsyncWrite, AsyncWriteExt, WriteHalf},
    sync::Mutex,
};
use tokio_serial::{DataBits, Parity, SerialStream, StopBits};

use self::response::Statuses;

use super::auv_control_board::{AUVControlBoard, MessageId};

pub mod response;

#[derive(Debug)]
pub struct MainElectronicsBoard<C: AsyncWrite + Unpin> {
    board: AUVControlBoard<C, Statuses>,
}

impl<C: AsyncWrite + Unpin> MainElectronicsBoard<C> {
    pub async fn new<T>(read_connection: T, write_connection: C) -> Self
    where
        T: 'static + AsyncReadExt + Unpin + Send,
    {
        Self {
            board: AUVControlBoard::new(
                Arc::new(Mutex::new(write_connection)),
                Statuses::new(read_connection).await,
                MessageId::default(),
            ),
        }
    }

    pub async fn serial(port_name: &str) -> Result<MainElectronicsBoard<WriteHalf<SerialStream>>> {
        const BAUD_RATE: u32 = 57600;
        const DATA_BITS: DataBits = DataBits::Eight;
        const PARITY: Parity = Parity::None;
        const STOP_BITS: StopBits = StopBits::One;

        let port_builder = tokio_serial::new(port_name, BAUD_RATE)
            .data_bits(DATA_BITS)
            .parity(PARITY)
            .stop_bits(STOP_BITS);
        let (read, write) = tokio::io::split(SerialStream::open(&port_builder)?);
        Ok(MainElectronicsBoard::<WriteHalf<SerialStream>>::new(read, write).await)
    }
}

impl<C: AsyncWrite + Unpin> MainElectronicsBoard<C> {
    pub async fn temperature(&self) -> Option<f32> {
        (*self.board.responses().temp().read().await).map(f32::from_le_bytes)
    }

    pub async fn humidity(&self) -> Option<f32> {
        (*self.board.responses().humid().read().await).map(f32::from_le_bytes)
    }

    pub async fn leak(&self) -> Option<bool> {
        *self.board.responses().leak().read().await
    }

    pub async fn thruster_arm(&self) -> Option<bool> {
        *self.board.responses().thruster_arm().read().await
    }

    pub async fn system_voltage(&self) -> Option<f32> {
        (*self.board.responses().system_voltage().read().await).map(f32::from_le_bytes)
    }

    pub async fn shutdown_cause(&self) -> Option<u8> {
        *self.board.responses().shutdown().read().await
    }
}



#[derive(Debug, Copy, Clone)]
pub enum MsbTaskId {
    T1Trig = 0x3,
    T2Trig = 0x4,
    D1Trig = 0x1,
    D2Trig = 0x2,
    Reset  = 0x0,
}

#[derive(Debug, Copy, Clone)]
pub enum MebCmd {
    Msb(MsbTaskId),
    Led{Index: char, R: u8, G: u8, B: u8},
    Led{Index: char, Color: LedColorHex},
}

#[derive(Debug, Copy, Clone)]
pub struct LedColorHex {
    r: u8,
    g: u8,
    b: u8,
}

pub const COLOR_BLK: LedColorHex = LedColorHex{r: 0x00, g: 0x00, b: 0x00};
pub const COLOR_WHT: LedColorHex = LedColorHex{r: 0xFF, g: 0xFF, b: 0xFF};
pub const COLOR_RED: LedColorHex = LedColorHex{r: 0xFF, g: 0x00, b: 0x00};
pub const COLOR_YLW: LedColorHex = LedColorHex{r: 0xF0, g: 0x80, b: 0x00};
pub const COLOR_GRN: LedColorHex = LedColorHex{r: 0x00, g: 0xFF, b: 0x00};
pub const COLOR_BLU: LedColorHex = LedColorHex{r: 0x00, g: 0x00, b: 0xFF};
pub const COLOR_MAG: LedColorHex = LedColorHex{r: 0xFF, g: 0x00, b: 0x5E};


impl<C: AsyncWriteExt + Unpin> MainElectronicsBoard<C> {
    pub async fn send_msg(&self, cmd: MebCmd) -> anyhow::Result<()> {
        let formatted_cmd = match MebCmd {
            Self::Msb(MsbTaskId) => {
                let formatted_cmd: [u8; 4] = [b'M', b'S', b'B', MsbTaskId as u8];
            }
            Self::Led{
                Index,
                R,
                G,
                B,
            } => {
                let formatted_cmd: [u8; 7] = [b'L', b'E', b'D', Index as u8, R, G, B];
            }
            Self::Led{Index, Color} => {
                let formatted_cmd: [u8; 7] = [b'L', b'E', b'D', Index as u8, Color.r, Color.g, Color.b];
            }
        };
        self.board.write_out_basic(formatted_cmd.to_vec()).await
    }
}
