use core::fmt::Debug;
use std::sync::Mutex;

use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio_serial::{DataBits, Parity, SerialStream, StopBits};

use self::{response::ResponseMap, util::Thruster};

pub mod response;
pub mod util;

const ID_LIMIT: u16 = 59999;

#[derive(Debug)]
struct MessageId {
    id: Mutex<u16>,
}

impl Default for MessageId {
    fn default() -> Self {
        MessageId { id: 0.into() }
    }
}

impl MessageId {
    async fn get(&mut self) -> u16 {
        let mut id = self.id.lock().unwrap();
        let ret = *id;
        *id += 1;
        if *id > ID_LIMIT {
            *id = 0
        }
        ret
    }
}

#[derive(Debug)]
pub struct ControlBoard<T>
where
    T: AsyncWriteExt + Unpin,
{
    comm_out: T,
    responses: ResponseMap,
    msg_id: MessageId,
}

const BAUD_RATE: u32 = 9600;
const DATA_BITS: DataBits = DataBits::Eight;
const PARITY: Parity = Parity::None;
const STOP_BITS: StopBits = StopBits::One;

impl ControlBoard<SerialStream> {
    async fn serial(port_name: &str) -> tokio_serial::Result<Self> {
        let port_builder = tokio_serial::new(port_name, BAUD_RATE)
            .data_bits(DATA_BITS)
            .parity(PARITY)
            .stop_bits(STOP_BITS);
        let responses = ResponseMap::new(SerialStream::open(&port_builder)?);
        let comm_out = SerialStream::open(&port_builder)?;

        Ok(Self {
            responses: responses.await,
            comm_out,
            msg_id: MessageId::default(),
        })
    }
}

impl<T: AsyncWrite + Unpin> ControlBoard<T> {
    /// Adds protocol requirements (e.g. message id, escapes) to a message body
    /// Returns the id assigned to the message
    fn add_metadata(&mut self, _message: &mut Vec<u8>) -> u16 {
        unimplemented!();
    }

    /// https://mb3hel.github.io/AUVControlBoard/user_guide/messages/#configuration-commands
    #[allow(clippy::too_many_arguments)]
    async fn motor_matrix_set(
        &mut self,
        thruster: Thruster,
        x: f32,
        y: f32,
        z: f32,
        pitch: f32,
        roll: f32,
        yaw: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Oversized to avoid reallocations
        let mut message: Vec<u8> = Vec::with_capacity(32 * 8);
        message.extend(thruster.to_be_bytes());
        message.extend(x.to_be_bytes());
        message.extend(y.to_be_bytes());
        message.extend(z.to_be_bytes());
        message.extend(pitch.to_be_bytes());
        message.extend(roll.to_be_bytes());
        message.extend(yaw.to_be_bytes());

        let id = self.add_metadata(&mut message);
        self.comm_out.write_all(&message).await?;
        // Spec guarantees empty response
        Ok(self.responses.get_ack(id).await.map(|_| ())?)
    }
}
