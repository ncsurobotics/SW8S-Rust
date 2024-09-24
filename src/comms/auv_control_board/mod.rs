use core::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use tokio::{io::AsyncWriteExt, sync::Mutex};

use self::util::{crc_itt16_false, AcknowledgeErr};

use super::auv_control_board::util::{END_BYTE, ESCAPE_BYTE, START_BYTE};

pub mod response;
pub mod util;

#[allow(async_fn_in_trait)]
pub trait GetAck {
    async fn get_ack(&self, id: u16) -> Result<Vec<u8>, AcknowledgeErr>;
}

const ID_LIMIT: u16 = 59999;

#[derive(Debug)]
pub struct MessageId {
    id: Mutex<u16>,
}

impl Default for MessageId {
    fn default() -> Self {
        MessageId { id: 0.into() }
    }
}

impl MessageId {
    pub async fn get(&self) -> u16 {
        let mut id = self.id.lock().await;
        let ret = *id;
        *id += 1;
        if *id > ID_LIMIT {
            *id = 0
        }
        ret
    }
}

#[derive(Debug)]
pub struct AUVControlBoard<T, U>
where
    T: AsyncWriteExt + Unpin,
    U: GetAck,
{
    comm_out: Arc<Mutex<T>>,
    responses: U,
    msg_id: MessageId,
}

impl<T: AsyncWriteExt + Unpin, U: GetAck> AUVControlBoard<T, U> {
    pub fn new(comm_out: Arc<Mutex<T>>, responses: U, msg_id: MessageId) -> Self {
        Self {
            comm_out,
            responses,
            msg_id,
        }
    }

    pub fn responses(&self) -> &U {
        &self.responses
    }

    /// Adds protocol requirements (e.g. message id, escapes) to a message body
    /// Returns the id assigned to the message and the message
    async fn add_metadata(&self, message: &[u8]) -> (u16, Vec<u8>) {
        let add_escape = |byte| {
            if [START_BYTE, END_BYTE, ESCAPE_BYTE].contains(&byte) {
                vec![ESCAPE_BYTE, byte]
            } else {
                vec![byte]
            }
        };

        let id = self.msg_id.get().await;

        let id_and_body: Vec<u8> = id
            .to_be_bytes()
            .into_iter()
            .chain(message.iter().copied())
            .collect();

        // Vecs isn't optimal, check if can reconcile one and two element arrays instead
        let mut formatted_message: Vec<u8> = Vec::from([START_BYTE]);
        formatted_message.extend(
            // Add escapes to id and message body
            id.to_be_bytes()
                .into_iter()
                .chain(message.iter().cloned())
                // Add CRC
                .chain(crc_itt16_false(&id_and_body).to_be_bytes().into_iter())
                .flat_map(add_escape),
        );
        formatted_message.push(END_BYTE);

        (id, formatted_message)
    }

    /// Writes out a message body and only gives acknowledge status
    /// Only for communications that return no data with acknowledge
    pub async fn write_out_basic(&self, message_body: Vec<u8>) -> Result<()> {
        let (id, message) = self.add_metadata(&message_body).await;
        self.comm_out.lock().await.write_all(&message).await?;
        // Spec guarantees empty response
        self.responses.get_ack(id).await?;
        Ok(())
    }

    /// Writes out a message body and only gives acknowledge status
    /// Only for communications that return no data with acknowledge
    pub async fn write_out(&self, message_body: Vec<u8>) -> Result<Vec<u8>> {
        let (id, message) = self.add_metadata(&message_body).await;
        self.comm_out.lock().await.write_all(&message).await?;
        // Spec guarantees empty response
        Ok(self.responses.get_ack(id).await?)
    }

    pub async fn write_out_no_response(&self, message_body: Vec<u8>) -> Result<()> {
        let (_, message) = self.add_metadata(&message_body).await;
        let mut comm_out = self.comm_out.lock().await;
        comm_out.write_all(&message).await?;
        comm_out.flush().await?;
        comm_out.shutdown().await?;
        Ok(())
    }
}
