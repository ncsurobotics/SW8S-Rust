use std::sync::{
    mpsc::{channel, Sender, TryRecvError},
    Arc,
};

use crate::comms::auv_control_board::{response::get_messages, util::crc_itt16_false};

use derive_getters::Getters;
use futures::{stream, StreamExt};
use tokio::{io::AsyncReadExt, sync::RwLock};

type Lock<T> = Arc<RwLock<Option<T>>>;

const AHT10: [u8; 5] = *b"AHT10";
const LEAK: [u8; 4] = *b"LEAK";
const TARM: [u8; 4] = *b"TARM";
const VSYS: [u8; 4] = *b"VSYS";
const SDOWN: [u8; 5] = *b"SDOWN";

#[derive(Debug, Getters)]
pub struct Statuses {
    aht10: Lock<[u8; 4 * 2]>,
    leak: Lock<bool>,
    thruster_arm: Lock<bool>,
    system_voltage: Lock<[u8; 4]>,
    shutdown: Lock<u8>,
    _tx: Sender<()>,
}

// Completely arbitrary
const DEFAULT_BUF_LEN: usize = 512;

impl Statuses {
    pub async fn new<T>(read_connection: T) -> Self
    where
        T: 'static + AsyncReadExt + Unpin + Send,
    {
        let aht10: Lock<_> = Arc::default();
        let leak: Lock<_> = Arc::default();
        let thruster_arm: Lock<_> = Arc::default();
        let system_voltage: Lock<_> = Arc::default();
        let shutdown: Lock<_> = Arc::default();
        let (_tx, rx) = channel::<()>(); // Signals struct destruction to thread
                                         //
        let aht10_clone = aht10.clone();
        let leak_clone = leak.clone();
        let thruster_arm_clone = thruster_arm.clone();
        let system_voltage_clone = system_voltage.clone();
        let shutdown_clone = shutdown.clone();

        tokio::spawn(async move {
            let mut buffer = Vec::with_capacity(DEFAULT_BUF_LEN);
            let mut serial_conn = read_connection;

            while rx.try_recv() != Err(TryRecvError::Disconnected) {
                Self::update_status(
                    &mut buffer,
                    &mut serial_conn,
                    &aht10_clone,
                    &leak_clone,
                    &thruster_arm_clone,
                    &system_voltage_clone,
                    &shutdown_clone,
                )
                .await;
            }
        });

        Self {
            aht10,
            leak,
            thruster_arm,
            system_voltage,
            shutdown,
            _tx,
        }
    }
}

impl Statuses {
    async fn update_status<T>(
        buffer: &mut Vec<u8>,
        serial_conn: &mut T,
        aht10: &RwLock<Option<[u8; 4 * 2]>>,
        leak: &RwLock<Option<bool>>,
        tarm: &RwLock<Option<bool>>,
        vsys: &RwLock<Option<[u8; 4]>>,
        sdown: &RwLock<Option<u8>>,
    ) where
        T: AsyncReadExt + Unpin + Send,
    {
        stream::iter(get_messages(buffer, serial_conn).await).for_each_concurrent(None, |message| async move {
            let id = u16::from_be_bytes(message[0..2].try_into().unwrap());
            let message_body = &message[2..(message.len() - 2)];
            let payload = &message[0..(message.len() - 2)];
            let given_crc =
                u16::from_be_bytes(message[(message.len() - 2)..].try_into().unwrap());
            let calculated_crc = crc_itt16_false(payload);

            if given_crc == calculated_crc {
                if message_body[0..5] == AHT10 {
                    *aht10.write().await = Some(message_body[5..].try_into().unwrap());
                } else if message_body[0..4] == LEAK {
                    *leak.write().await = Some(message_body[4] == 1);
                } else if message_body[0..4] == TARM {
                    *tarm.write().await = Some(message_body[4] == 1);
                } else if message_body[0..4] == VSYS {
                    *vsys.write().await = Some(message_body[4..].try_into().unwrap());
                } else if message_body[0..4] == SDOWN {
                    *sdown.write().await = Some(message_body[4]);
                } else {
                    eprintln!("Unknown MEB message (id: {id}) {:?}", message_body);
                }
            } else {
                eprintln!(
                "Given CRC ({given_crc}) != calculated CRC ({calculated_crc}) for MEB message (id: {id}) {:?}",
                message_body
            );
            }
        }).await;
    }
}
