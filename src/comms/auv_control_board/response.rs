use std::time::Duration;

use tokio::{io::AsyncReadExt, time::sleep};

use super::util::{END_BYTE, ESCAPE_BYTE, START_BYTE};

pub fn find_end(buffer: &[u8]) -> Option<(usize, &u8)> {
    buffer
        .iter()
        .enumerate()
        .skip(1)
        .find(|(idx, val)| **val == END_BYTE && buffer[idx - 1] != ESCAPE_BYTE)
}

pub fn check_start(buffer: &mut Vec<u8>, end_idx: usize) -> bool {
    // Adjust for starting without start byte (malformed comms)
    // TODO: log feature for these events -- serious issues!!!
    match buffer
        .iter()
        .enumerate()
        .find(|(idx, val)| **val == START_BYTE && buffer[idx - 1] != ESCAPE_BYTE)
    {
        Some((0, _)) => true, // Expected condition
        None => {
            eprintln!(
                "Buffer has end byte but no start byte, discarding {:?}",
                &buffer[0..=end_idx]
            );
            buffer.drain(0..=end_idx);
            false // Escape and try again on next value
        }
        Some((start_idx, _)) => {
            eprintln!(
                "Buffer does not begin with start byte, discarding {:?}",
                &buffer[0..start_idx]
            );
            buffer.drain(0..start_idx);
            true
        }
    }
}

/// Discard start, end, and escape bytes
pub fn clean_message(buffer: &mut Vec<u8>, end_idx: usize) -> Vec<u8> {
    let message: Vec<_> = buffer
        .drain(0..=end_idx)
        .skip(1)
        .filter(|&byte| byte != ESCAPE_BYTE)
        .collect();
    message[0..message.len() - 1].to_vec()
}

/// Reads from serial resource, updating ack_map
pub async fn get_messages<T>(buffer: &mut Vec<u8>, serial_conn: &mut T) -> Vec<Vec<u8>>
where
    T: AsyncReadExt + Unpin,
{
    let buf_len = buffer.len();
    // Read bytes up to buffer capacity
    let _ = serial_conn.read(&mut buffer[buf_len..]).await.unwrap();
    let mut messages = Vec::new();

    while let Some((end_idx, _)) = find_end(buffer) {
        if !check_start(buffer, end_idx) {
            continue;
        };
        messages.push(clean_message(buffer, end_idx));
    }

    println!("Unprocessed Length: {}", buffer.len());
    messages
}
