use std::{
    collections::HashMap,
    sync::{
        mpsc::{channel, Sender, TryRecvError},
        Arc,
    },
    time::{Duration, SystemTime},
};

use derive_getters::Getters;
use futures::stream;
use futures::StreamExt;
use tokio::{
    io::{stderr, AsyncReadExt, AsyncWriteExt},
    sync::{Mutex, RwLock},
    time::sleep,
};

use crate::{
    comms::auv_control_board::{response::get_messages, util::crc_itt16_false_bitmath, GetAck},
    write_stream_mutexed,
};

use crate::comms::auv_control_board::util::AcknowledgeErr;

use super::util::Angles;

const ACK: [u8; 3] = *b"ACK";
const WDGS: [u8; 4] = *b"WDGS";
const BNO055D: [u8; 7] = *b"BNO055D";
const MS5837D: [u8; 7] = *b"MS5837D";
#[allow(dead_code)]
const DEBUG: [u8; 5] = *b"DEBUG";
#[allow(dead_code)]
const DBGDAT: [u8; 6] = *b"DBGDAT";

pub type KeyedAcknowledges = HashMap<u16, Result<Vec<u8>, AcknowledgeErr>>;

#[derive(Debug, Getters)]
pub struct ResponseMap {
    ack_map: Arc<Mutex<KeyedAcknowledges>>,
    watchdog_status: Arc<RwLock<Option<bool>>>,
    bno055_status: Arc<RwLock<Option<[u8; 4 * 7]>>>,
    ms5837_status: Arc<RwLock<Option<[u8; 4 * 3]>>>,
    _tx: Sender<()>,
}

// Completely arbitrary
const DEFAULT_BUF_LEN: usize = 512;
pub const MAP_POLL_SLEEP: Duration = Duration::from_millis(5);

impl ResponseMap {
    pub async fn new<T>(read_connection: T) -> Self
    where
        T: 'static + AsyncReadExt + Unpin + Send,
    {
        let ack_map: Arc<Mutex<_>> = Arc::default();
        let watchdog_status: Arc<RwLock<_>> = Arc::default();
        let bno055_status: Arc<RwLock<_>> = Arc::default();
        let ms5837_status: Arc<RwLock<_>> = Arc::default();
        let (_tx, rx) = channel::<()>(); // Signals struct destruction to thread

        // Independent thread that live updates maps forever
        let ack_map_clone = ack_map.clone();
        let watchdog_status_clone = watchdog_status.clone();
        let bno055_status_clone = bno055_status.clone();
        let ms5837_status_clone = ms5837_status.clone();

        tokio::spawn(async move {
            let mut buffer = Vec::with_capacity(DEFAULT_BUF_LEN);
            let mut serial_conn = read_connection;

            while rx.try_recv() != Err(TryRecvError::Disconnected) {
                Self::update_maps(
                    &mut buffer,
                    &mut serial_conn,
                    &ack_map_clone,
                    &watchdog_status_clone,
                    &bno055_status_clone,
                    &ms5837_status_clone,
                    &mut stderr(),
                )
                .await;
            }
        });

        Self {
            ack_map,
            watchdog_status,
            bno055_status,
            ms5837_status,
            _tx,
        }
    }

    /// Reads from serial resource, updating ack_map
    pub async fn update_maps<T, U>(
        buffer: &mut Vec<u8>,
        serial_conn: &mut T,
        ack_map: &Mutex<KeyedAcknowledges>,
        watchdog_status: &RwLock<Option<bool>>,
        bno055_status: &RwLock<Option<[u8; 4 * 7]>>,
        ms5837_status: &RwLock<Option<[u8; 4 * 3]>>,
        err_stream: &mut U,
    ) where
        T: AsyncReadExt + Unpin + Send,
        U: AsyncWriteExt + Unpin + Send,
    {
        let err_stream = &Mutex::new(err_stream);
        stream::iter(get_messages(buffer, serial_conn, #[cfg(feature = "logging")] "control_board_in").await).for_each_concurrent(None, |message| async move {
            let id = u16::from_be_bytes(message[0..2].try_into().unwrap());
            let message_body = &message[2..(message.len() - 2)];
            let payload = &message[0..(message.len() - 2)];
            let given_crc = u16::from_be_bytes(message[(message.len() - 2)..].try_into().unwrap());
            let calculated_crc = crc_itt16_false_bitmath(payload);

            if given_crc == calculated_crc {
                if message_body.get(0..3) == Some(&ACK) {
                    let id = u16::from_be_bytes(message_body[3..=4].try_into().unwrap());
                    let error_code: u8 = message_body[5];

                    let val = if error_code == 0 {
                        Ok(message_body[6..].to_vec())
                    } else {
                        Err(AcknowledgeErr::from(error_code))
                    };
                    ack_map.lock().await.insert(id, val);
                } else if message_body.get(0..4) == Some(&WDGS) {
                    *watchdog_status.write().await = Some(message_body[4] != 0);
                } else if message_body.get(0..7) == Some(&BNO055D) {
                    static mut PREV_YAW_PRINT: SystemTime = SystemTime::UNIX_EPOCH;
                    let new_status = message_body[7..].try_into().unwrap();
                    
                    let now = SystemTime::now();
                    unsafe {
                        if now.duration_since(PREV_YAW_PRINT).unwrap() > Duration::from_secs(1) {
                            logln!("Current yaw reading: {}", 
                        Angles::from_raw(new_status).yaw()
                                );
                        PREV_YAW_PRINT = SystemTime::now();
                        }
                    }
                   

                    *bno055_status.write().await = Some(new_status);
                } else if message_body.get(0..7) == Some(&MS5837D) {
                    *ms5837_status.write().await = Some(message_body[7..].try_into().unwrap());
                } else {
                    write_stream_mutexed!(err_stream, format!("Unknown message (id: {id}) {:?}\n", payload));
                }
            } else {
                write_stream_mutexed!(err_stream,
                format!(
                "Given CRC ({given_crc} {:?}) != calculated CRC ({calculated_crc} {:?}) for message (id: {id}) {:?} (0x{})\n",
                given_crc.to_ne_bytes(),
                calculated_crc.to_ne_bytes(),
                payload,
                payload.iter().map(|byte| format!("{byte:02x}").to_string()).reduce(|acc, x| acc + &x).unwrap_or("".to_string())
            ));
            }
        }).await
    }

    pub async fn get_angles(&self) -> Option<Angles> {
        (*self.bno055_status.read().await).map(Angles::from_raw)
    }
}

impl GetAck for ResponseMap {
    async fn get_ack(&self, id: u16) -> Result<Vec<u8>, AcknowledgeErr> {
        loop {
            if let Some(x) = self.ack_map.lock().await.remove(&id) {
                return x;
            }
            sleep(MAP_POLL_SLEEP).await; // Allow for new data from serial
        }
    }
}
