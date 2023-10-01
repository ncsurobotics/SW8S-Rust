use std::{
    sync::{
        mpsc::{channel, Sender, TryRecvError},
        Arc,
    },
    time::Duration,
};

use crate::{
    comms::auv_control_board::{response::get_messages, util::crc_itt16_false_bitmath},
    write_stream_mutexed,
};

use derive_getters::Getters;
use futures::{stream, StreamExt};
use tokio::{
    io::{stderr, AsyncReadExt, AsyncWriteExt},
    sync::{Mutex, RwLock},
    time::sleep,
};

type Lock<T> = Arc<RwLock<Option<T>>>;

const AHT10: [u8; 5] = *b"AHT10";
const TEMP: [u8; 4] = *b"TEMP";
const LEAK: [u8; 4] = *b"LEAK";
const TARM: [u8; 4] = *b"TARM";
const VSYS: [u8; 4] = *b"VSYS";
const SDOWN: [u8; 5] = *b"SDOWN";

#[derive(Debug, Getters)]
pub struct Statuses {
    temp: Lock<[u8; 4]>,
    humid: Lock<[u8; 4]>,
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
        let temp: Lock<_> = Arc::default();
        let humid: Lock<_> = Arc::default();
        let leak: Lock<_> = Arc::default();
        let thruster_arm: Lock<_> = Arc::default();
        let system_voltage: Lock<_> = Arc::default();
        let shutdown: Lock<_> = Arc::default();
        let (_tx, rx) = channel::<()>(); // Signals struct destruction to thread
                                         //
        let temp_clone = temp.clone();
        let humid_clone = humid.clone();
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
                    &temp_clone,
                    &humid_clone,
                    &leak_clone,
                    &thruster_arm_clone,
                    &system_voltage_clone,
                    &shutdown_clone,
                    &mut stderr(),
                )
                .await;
            }
        });

        Self {
            temp,
            humid,
            leak,
            thruster_arm,
            system_voltage,
            shutdown,
            _tx,
        }
    }
}

impl Statuses {
    #[allow(clippy::too_many_arguments)]
    pub async fn update_status<T, U>(
        buffer: &mut Vec<u8>,
        serial_conn: &mut T,
        temp: &RwLock<Option<[u8; 4]>>,
        humid: &RwLock<Option<[u8; 4]>>,
        leak: &RwLock<Option<bool>>,
        tarm: &Arc<RwLock<Option<bool>>>,
        vsys: &RwLock<Option<[u8; 4]>>,
        sdown: &RwLock<Option<u8>>,
        err_stream: &mut U,
    ) where
        T: AsyncReadExt + Unpin + Send,
        U: AsyncWriteExt + Unpin + Send,
    {
        let err_stream = &Mutex::new(err_stream);
        let messages = get_messages(
            buffer,
            serial_conn,
            #[cfg(feature = "logging")]
            "meb_in",
        )
        .await;
        stream::iter(messages).for_each_concurrent(None, |message| async move {
            if message.len() < 4 { println!("Message len < 4: {:?}", message); return; };

            let id = u16::from_be_bytes(message[0..2].try_into().unwrap());
            let message_body = &message[2..(message.len() - 2)];
            let payload = &message[0..(message.len() - 2)];
            let given_crc =
                u16::from_be_bytes(message[(message.len() - 2)..].try_into().unwrap());
            let calculated_crc = crc_itt16_false_bitmath(payload);

            if given_crc == calculated_crc {
                if message_body.get(0..5) == Some(&AHT10) {
                    *temp.write().await = Some(message_body[5..9].try_into().unwrap());
                    *humid.write().await = Some(message_body[(5 + 4)..].try_into().unwrap());
                } else if message_body.get(0..4) == Some(&TEMP) {
                    *temp.write().await = Some(message_body[4..8].try_into().unwrap());
                    *humid.write().await = Some(message_body[(4 + 4)..].try_into().unwrap());
                } else if message_body.get(0..4) == Some(&LEAK) {
                    *leak.write().await = Some(message_body[4] == 1);
                } else if message_body.get(0..4) == Some(&TARM) {
                    let tarm_clone = tarm.clone();
                    let tarm_status = Some(message_body[4] == 1);
                    tokio::spawn(async move {
                        sleep(Duration::from_millis(3500)).await;
                        *tarm_clone.write().await = tarm_status;
                    });
                } else if message_body.get(0..4) == Some(&VSYS) {
                    *vsys.write().await = Some(message_body[4..].try_into().unwrap());
                } else if message_body.get(0..4) == Some(&SDOWN) {
                    *sdown.write().await = Some(message_body[4]);
                } else {
                    write_stream_mutexed!(err_stream, format!("Unknown MEB message (id: {id}) {:?}\n", message_body));
                }
            } else {
                write_stream_mutexed!(err_stream, format!(
                "Given CRC ({given_crc} {:?}) != calculated CRC ({calculated_crc} {:?}) for message (id: {id}) {:?} (0x{})\n",
                given_crc.to_ne_bytes(),
                calculated_crc.to_ne_bytes(),
                message_body,
                message_body.iter().map(|byte| format!("{:02x}", byte).to_string()).reduce(|acc, x| acc + &x).unwrap()
            ));
            }
        }).await;
    }
}
