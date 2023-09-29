use self::response::Statuses;

use anyhow::Result;
use tokio::io::AsyncReadExt;
use tokio_serial::{DataBits, Parity, SerialStream, StopBits};

mod response;

#[derive(Debug)]
pub struct MainElectronicsBoard {
    statuses: Statuses,
}

impl MainElectronicsBoard {
    pub async fn new<T>(read_connection: T) -> Self
    where
        T: 'static + AsyncReadExt + Unpin + Send,
    {
        Self {
            statuses: Statuses::new(read_connection).await,
        }
    }

    pub async fn serial(port_name: &str) -> Result<Self> {
        const BAUD_RATE: u32 = 57600;
        const DATA_BITS: DataBits = DataBits::Eight;
        const PARITY: Parity = Parity::None;
        const STOP_BITS: StopBits = StopBits::One;

        let port_builder = tokio_serial::new(port_name, BAUD_RATE)
            .data_bits(DATA_BITS)
            .parity(PARITY)
            .stop_bits(STOP_BITS);
        Ok(Self::new(SerialStream::open(&port_builder)?).await)
    }
}

impl MainElectronicsBoard {
    pub async fn temperature(&self) -> Option<f32> {
        (*self.statuses.aht10().read().await)
            .map(|aht10| f32::from_le_bytes(aht10[0..4].try_into().unwrap()))
    }

    pub async fn humidity(&self) -> Option<f32> {
        (*self.statuses.aht10().read().await)
            .map(|aht10| f32::from_le_bytes(aht10[4..].try_into().unwrap()))
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
