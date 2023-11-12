use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use itertools::Itertools;
use tokio::io::AsyncReadExt;
use tokio_serial::{DataBits, Parity, SerialStream, StopBits};

use self::response::Statuses;

pub mod response;

#[derive(Debug)]
pub struct MainElectronicsBoard {
    statuses: Statuses,
    arm_count: Arc<Mutex<Vec<bool>>>,
    arm_state: Arc<Mutex<Option<bool>>>,
}

impl MainElectronicsBoard {
    pub async fn new<T>(read_connection: T) -> Self
        where
            T: 'static + AsyncReadExt + Unpin + Send,
    {
        Self {
            statuses: Statuses::new(read_connection).await,
            arm_count: Arc::new(Mutex::new(vec![false; 24])),
            arm_state: Arc::new(Mutex::new(Some(false))),
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
        (*self.statuses.temp().read().await).map(f32::from_le_bytes)
    }

    pub async fn humidity(&self) -> Option<f32> {
        (*self.statuses.humid().read().await).map(f32::from_le_bytes)
    }

    pub async fn leak(&self) -> Option<bool> {
        *self.statuses.leak().read().await
    }

    pub async fn thruster_arm(&self) -> Option<bool> {
        let arm_count = self.arm_count.clone();
        let arm_state = self.arm_state.clone();
        let current_arm_state = self.arm_state.lock().unwrap().clone();

        let t1 = thread::spawn(move || {
            let mut locked_arm_count = arm_count.lock().unwrap();
            let mut locked_arm_state = arm_state.lock().unwrap();

            locked_arm_count.push(current_arm_state.unwrap_or(false));
            locked_arm_count.remove(0);

            if locked_arm_count.iter().all_equal() {
                *locked_arm_state = current_arm_state;
            }
        });

        t1.join().unwrap();
        self.arm_state.lock().unwrap().clone()
    }

    pub async fn system_voltage(&self) -> Option<f32> {
        (*self.statuses.system_voltage().read().await).map(f32::from_le_bytes)
    }

    pub async fn shutdown_cause(&self) -> Option<u8> {
        *self.statuses.shutdown().read().await
    }
}
