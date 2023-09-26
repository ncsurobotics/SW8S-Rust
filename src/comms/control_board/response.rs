use std::{
    collections::HashMap,
    sync::{
        mpsc::{channel, Sender, TryRecvError},
        Arc,
    },
    time::Duration,
};

use tokio::{io::AsyncReadExt, sync::Mutex, time::sleep};

use super::util::{crc, AcknowledgeErr, END_BYTE, ESCAPE_BYTE, START_BYTE};

const ACK: [u8; 3] = *b"ACK";
const WDGS: [u8; 4] = *b"WDGS";
const BNO055D: [u8; 7] = *b"BNO055D";
const MS5837D: [u8; 7] = *b"MS5837D";
const DEBUG: [u8; 5] = *b"DEBUG";
const DBGDAT: [u8; 6] = *b"DBGDAT";

#[derive(Debug)]
pub struct ResponseMap {
    ack_map: Arc<Mutex<HashMap<u16, Result<Vec<u8>, AcknowledgeErr>>>>,
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

    pub async fn get_ack(&mut self, id: u16) -> Result<Vec<u8>, AcknowledgeErr> {
        loop {
            if let Some(x) = self.ack_map.lock().await.remove(&id) {
                return x;
            }
            sleep(MAP_POLL_SLEEP).await; // Allow for new data from serial
        }
    }

    /// Reads from serial resource, updating ack_map
    async fn update_maps<T>(
        buffer: &mut Vec<u8>,
        serial_conn: &mut T,
        ack_map: &Mutex<HashMap<u16, Result<Vec<u8>, AcknowledgeErr>>>,
        watchdog_status: &Mutex<Option<bool>>,
        bno055_status: &Mutex<Option<[u8; 8 * 7]>>,
        ms5837_status: &Mutex<Option<[u8; 8 * 3]>>,
    ) where
        T: AsyncReadExt + Unpin,
    {
        let buf_len = buffer.len();
        // Read bytes up to buffer capacity
        let _ = serial_conn.read(&mut buffer[buf_len..]).await.unwrap();

        while let Some(end) = buffer
            .iter()
            .enumerate()
            .skip(1)
            .find(|(idx, val)| **val == END_BYTE && buffer[idx - 1] != ESCAPE_BYTE)
        {
            let mut end_idx = end.0;

            // Adjust for starting without start byte (malformed comms)
            // TODO: log feature for these events -- serious issues!!!
            match buffer
                .iter()
                .enumerate()
                .find(|(idx, val)| **val == START_BYTE && buffer[idx - 1] != ESCAPE_BYTE)
            {
                Some((0, _)) => (), // Expected condition
                None => {
                    eprintln!(
                        "Buffer has end byte but no start byte, discarding {:?}",
                        &buffer[0..=end_idx]
                    );
                    buffer.drain(0..=end_idx);
                    continue; // Escape and try again on next value
                }
                Some((start_idx, _)) => {
                    eprintln!(
                        "Buffer does not begin with start byte, discarding {:?}",
                        &buffer[0..start_idx]
                    );
                    buffer.drain(0..start_idx);

                    if end_idx < start_idx {
                        eprintln!(
                            "First buffer start byte is behind end byte, discarding {:?}",
                            &buffer[0..start_idx]
                        );
                        buffer.drain(0..start_idx);
                        continue; // Escape and try again on next value
                    } else {
                        // end_idx > x, end_idx == x is impossible
                        end_idx -= start_idx;
                    }
                }
            };

            // Discard start, end, and escape bytes
            let message: Vec<_> = buffer
                .drain(0..=end_idx)
                .skip(1)
                .filter(|&byte| byte != ESCAPE_BYTE)
                .collect();
            let message = &message[0..message.len() - 1];

            let id = u16::from_be_bytes(message[0..2].try_into().unwrap());
            let message_body = &message[2..(message.len() - 2)];
            let payload = &message[0..(message.len() - 2)];
            let given_crc = u16::from_be_bytes(message[(message.len() - 2)..].try_into().unwrap());
            let calculated_crc = crc(payload);

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
                    *watchdog_status.lock().await = Some(message_body[5] != 0);
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
        }
    }
}
