use tokio::io::AsyncReadExt;

use super::util::{END_BYTE, ESCAPE_BYTE, START_BYTE};

pub fn find_end(buffer: &[u8]) -> Option<(usize, &u8)> {
    buffer
        .iter()
        .enumerate()
        .skip(1)
        .find(|(idx, val)| **val == END_BYTE && buffer[idx - 1] != ESCAPE_BYTE)
}

/// Returns adjust end_idx
pub fn check_start(buffer: &mut Vec<u8>, end_idx: usize) -> Option<usize> {
    // Adjust for starting without start byte (malformed comms)
    // TODO: log feature for these events -- serious issues!!!
    match buffer
        .iter()
        .enumerate()
        .find(|(_, val)| **val == START_BYTE)
    {
        Some((0, _)) => Some(end_idx), // Expected condition
        None => {
            eprintln!(
                "Buffer has end byte but no start byte, discarding {:?}",
                &buffer[0..=end_idx]
            );
            buffer.drain(0..=end_idx);
            None // Escape and try again on next value
        }
        Some((start_idx, _)) => {
            if buffer[start_idx - 1] == ESCAPE_BYTE {
                eprintln!(
                    "First start byte in buffer was escaped, discarding {:?}",
                    &buffer[0..=start_idx]
                );
                buffer.drain(0..=start_idx);
                None
            } else {
                eprintln!(
                    "Buffer does not begin with start byte, discarding {:?}",
                    &buffer[0..start_idx]
                );
                buffer.drain(0..start_idx);
                Some(end_idx - start_idx)
            }
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
    T: AsyncReadExt + Unpin + Send,
{
    while find_end(buffer).is_none() {
        buffer.push(serial_conn.read_u8().await.unwrap());
    }
    // Read bytes up to buffer capacity
    let mut messages = Vec::new();
    println!("CURRENT BUFFER: {:?}", buffer);

    while let Some((end_idx, _)) = find_end(buffer) {
        match check_start(buffer, end_idx) {
            None => {
                println!("Failed start check");
            }
            Some(end_idx) => {
                println!("Passed start check");
                messages.push(clean_message(buffer, end_idx));
            }
        }
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    #[tokio::test]
    async fn start_not_at_front() {
        let input: Vec<u8> = vec![0, 1, START_BYTE, END_BYTE];
        let input2: Vec<u8> = vec![END_BYTE, 1, START_BYTE, 3, END_BYTE, 5];
        let mut buffer: Vec<u8> = Vec::with_capacity(512);

        assert_eq!(
            stream::iter(get_messages(&mut buffer, &mut &*input).await)
                .collect::<Vec<Vec<u8>>>()
                .await,
            vec![vec![]]
        );

        assert_eq!(
            stream::iter(get_messages(&mut buffer, &mut &*input2).await)
                .collect::<Vec<Vec<u8>>>()
                .await,
            vec![vec![3]]
        );
    }
}
