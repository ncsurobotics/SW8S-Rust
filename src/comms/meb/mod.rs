use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::io::ReadHalf;
use tokio::io::{AsyncReadExt, AsyncWriteExt, WriteHalf};
use tokio_serial::{DataBits, Parity, SerialStream, StopBits};

use self::response::Statuses;

pub mod response;

#[derive(Debug)]
pub struct MainElectronicsBoard<C> {
    comm_out: Arc<Mutex<C>>,
    statuses: Statuses,
}

impl<C> MainElectronicsBoard<C> {
    pub async fn new<T>(read_connection: T, write_connection: C) -> Self
    where
        T: 'static + AsyncReadExt + Unpin + Send,
        C: 'static + AsyncWriteExt + Unpin + Send,
    {
        Self {
            comm_out: Arc::new(Mutex::new(write_connection)),
            statuses: Statuses::new(read_connection).await,
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

impl<C> MainElectronicsBoard<C> {
    pub async fn temperature(&self) -> Option<f32> {
        (*self.statuses.temp().read().await).map(f32::from_le_bytes)
    }

    pub async fn humidity(&self) -> Option<f32> {
        (*self.statuses.humid().read().await).map(f32::from_le_bytes)
    }

    pub async fn leak(&self) -> Option<bool> {
        *self.statuses.leak().read().await
    }

    pub async fn thruster_arm(&self) -> Option<bool> {
        *self.statuses.thruster_arm().read().await
    }

    pub async fn system_voltage(&self) -> Option<f32> {
        (*self.statuses.system_voltage().read().await).map(f32::from_le_bytes)
    }

    pub async fn shutdown_cause(&self) -> Option<u8> {
        *self.statuses.shutdown().read().await
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MebCmd {
    T1Trig = 0x3,
    T2Trig = 0x4,
    D1Trig = 0x1,
    D2Trig = 0x2,
    Reset = 0x0,
}

impl<C: AsyncWriteExt + Unpin> MainElectronicsBoard<C> {
    #[allow(clippy::await_holding_lock)]
    pub async fn send_msg(&self, cmd: MebCmd) -> std::io::Result<()> {
        let formatted_cmd: [u8; 4] = [b'M', b'S', b'B', cmd as u8];
        let mut comm_out = self.comm_out.lock().unwrap();
        comm_out.write_all(&formatted_cmd).await
    }
}
