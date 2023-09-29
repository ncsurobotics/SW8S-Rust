use std::{
    collections::HashMap,
    sync::{
        mpsc::{channel, Sender, TryRecvError},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use futures::stream;
use futures::StreamExt;
use tokio::{io::AsyncReadExt, sync::Mutex, time::sleep};

use crate::comms::auv_control_board::{response::get_messages, util::crc_itt16_false, GetAck};

use crate::comms::auv_control_board::util::AcknowledgeErr;

const ACK: [u8; 3] = *b"ACK";
const WDGS: [u8; 4] = *b"WDGS";
const BNO055D: [u8; 7] = *b"BNO055D";
const MS5837D: [u8; 7] = *b"MS5837D";
const DEBUG: [u8; 5] = *b"DEBUG";
const DBGDAT: [u8; 6] = *b"DBGDAT";

type KeyedAcknowledges = HashMap<u16, Result<Vec<u8>, AcknowledgeErr>>;

#[derive(Debug)]
pub struct ResponseMap {
    ack_map: Arc<Mutex<KeyedAcknowledges>>,
    watchdog_status: Arc<Mutex<Option<bool>>>,
    bno055_status: Arc<Mutex<Option<[u8; 8 * 7]>>>,
    ms5837_status: Arc<Mutex<Option<[u8; 8 * 3]>>>,
    _tx: Sender<()>,
}

// Completely arbitrary
const DEFAULT_BUF_LEN: usize = 512;
const MAP_POLL_SLEEP: Duration = Duration::from_millis(5);

impl ResponseMap {
    pub async fn new<T>(read_connection: T) -> Self
    where
        T: 'static + AsyncReadExt + Unpin + Send,
    {
        let ack_map: Arc<Mutex<_>> = Arc::default();
        let watchdog_status: Arc<Mutex<_>> = Arc::default();
        let bno055_status: Arc<Mutex<_>> = Arc::default();
        let ms5837_status: Arc<Mutex<_>> = Arc::default();
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
    async fn update_maps<T>(
        buffer: &mut Vec<u8>,
        serial_conn: &mut T,
        ack_map: &Mutex<KeyedAcknowledges>,
        watchdog_status: &Mutex<Option<bool>>,
        bno055_status: &Mutex<Option<[u8; 8 * 7]>>,
        ms5837_status: &Mutex<Option<[u8; 8 * 3]>>,
    ) where
        T: AsyncReadExt + Unpin,
    {
        stream::iter(get_messages(buffer, serial_conn).await).for_each_concurrent(None, |message| async move {
            let id = u16::from_be_bytes(message[0..2].try_into().unwrap());
            let message_body = &message[2..(message.len() - 2)];
            let payload = &message[0..(message.len() - 2)];
            let given_crc = u16::from_be_bytes(message[(message.len() - 2)..].try_into().unwrap());
            let calculated_crc = crc_itt16_false(payload);

            if given_crc == calculated_crc {
                if message_body[0..3] == ACK {
                    let id = u16::from_be_bytes(message_body[3..=4].try_into().unwrap());
                    let error_code: u8 = message_body[5];

                    let val = if error_code == 0 {
                        Ok(message_body[6..].to_vec())
                    } else {
                        Err(AcknowledgeErr::from(error_code))
                    };
                    ack_map.lock().await.insert(id, val);
                } else if message_body[0..4] == WDGS {
                    *watchdog_status.lock().await = Some(message_body[4] != 0);
                } else if message_body[0..7] == BNO055D {
                    *bno055_status.lock().await = Some(message_body[7..].try_into().unwrap());
                } else if message_body[0..7] == MS5837D {
                    *ms5837_status.lock().await = Some(message_body[7..].try_into().unwrap());
                } else {
                    eprintln!("Unknown message (id: {id}) {:?}", message_body);
                }
            } else {
                eprintln!(
                "Given CRC ({given_crc}) != calculated CRC ({calculated_crc}) for message (id: {id}) {:?}",
                message_body
            );
            }
        }).await
    }
}

#[async_trait]
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
