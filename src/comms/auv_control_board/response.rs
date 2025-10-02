use bytes::BufMut;
use tokio::io::AsyncReadExt;

#[cfg(feature = "logging")]
use tokio::{fs::OpenOptions, io::AsyncWriteExt, sync::Mutex};

use super::util::{END_BYTE, ESCAPE_BYTE, START_BYTE};
use crate::logln;

#[cfg(feature = "logging")]
static LOG_NAMES: Mutex<Vec<String>> = Mutex::const_new(Vec::new());

pub fn find_end(buffer: &[u8]) -> Option<(usize, &u8)> {
    let mut prev_escaped = false;
    buffer.iter().enumerate().skip(1).find(|(_, byte)| {
        let ret = **byte == END_BYTE && !prev_escaped;
        prev_escaped = !prev_escaped && **byte == ESCAPE_BYTE;
        ret
    })
}

/// Returns adjust end_idx
pub fn check_start(buffer: &mut Vec<u8>, end_idx: usize) -> Option<usize> {
    // Adjust for starting without start byte (malformed comms)
    // TODO: log feature for these events -- serious issues!!!
    match buffer
        .iter()
        .enumerate()
        .find(|(idx, val)| **val == START_BYTE && (*idx == 0 || buffer[idx - 1] != ESCAPE_BYTE))
    {
        Some((0, _)) => Some(end_idx), // Expected condition
        None => {
            logln!(
                "Buffer has end byte but no start byte, discarding {:?}",
                &buffer[0..=end_idx]
            );
            buffer.drain(0..=end_idx);
            None // Escape and try again on next value
        }
        Some((start_idx, _)) => {
            if buffer[start_idx - 1] == ESCAPE_BYTE {
                logln!(
                    "First start byte in buffer was escaped, discarding {:?}",
                    &buffer[0..=start_idx]
                );
                buffer.drain(0..=start_idx);
                None
            } else {
                logln!(
                    "Buffer does not begin with start byte, discarding {:?}",
                    &buffer[0..start_idx]
                );
                buffer.drain(0..start_idx);
                if end_idx >= start_idx {
                    Some(end_idx - start_idx)
                } else {
                    None
                }
            }
        }
    }
}

/// Discard start, end, and escape bytes
pub fn clean_message(buffer: &mut Vec<u8>, end_idx: usize) -> Vec<u8> {
    let message: Vec<u8> = buffer.drain(0..=end_idx).collect();

    let mut prev_escaped = false;
    let message: Vec<_> = message
        .clone()
        .into_iter()
        .skip(1)
        .filter(|byte| {
            let ret = *byte != ESCAPE_BYTE || prev_escaped;
            prev_escaped = !prev_escaped && *byte == ESCAPE_BYTE;
            ret
        })
        .collect();
    message[0..message.len() - 1].to_vec()
}

/// Reads from serial resource, updating ack_map
pub async fn get_messages<T>(
    buffer: &mut Vec<u8>,
    serial_conn: &mut T,
    #[cfg(feature = "logging")] dump_file: &str,
) -> Vec<Vec<u8>>
where
    T: AsyncReadExt + Unpin + Send,
{
    if serial_conn.read_buf(buffer).await.unwrap() != 0 {
        let mut messages = Vec::new();

        // TODO fix order of messages with unblocked logging
        /*
        #[cfg(feature = "unblocked_logging")]
        {
            let buffer = buffer.clone();
            let dump_file = dump_file.to_string();
            tokio::spawn(
                async move {
                    write_log(&[buffer], &dump_file).await;
                }
            );
        }
        */

        #[cfg(all(feature = "logging"))]
        {
            write_log(&[buffer.clone()], dump_file).await;
        }

        while let Some((end_idx, _)) = find_end(buffer) {
            if let Some(end_idx) = check_start(buffer, end_idx) {
                messages.push(clean_message(buffer, end_idx));
            }
        }

        messages
    } else if buffer.has_remaining_mut() {
        Vec::new()
    } else {
        panic!("Buffer capacity filled!");
    }
}

#[cfg(feature = "logging")]
pub async fn write_log(messages: &[Vec<u8>], #[cfg(feature = "logging")] dump_file: &str) {
    if !std::path::Path::new("logging").exists() {
        std::fs::create_dir("logging").unwrap();
    }

    let file_dir = fmt_filename_time(dump_file).await;

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_dir)
        .await
    {
        for msg in messages.iter() {
            file.write_all(msg).await.unwrap()
        }

        file.flush().await.unwrap();
    } else {
        logln!("ERROR OPENING FILE IN LOGGING");
    }
}

#[cfg(feature = "logging")]
pub async fn fmt_filename_time(dump_file: &str) -> String {
    use crate::TIMESTAMP;

    let formatted_time = &*TIMESTAMP;

    let mut names = LOG_NAMES.lock().await;
    if !names.iter().any(|n| n == dump_file) {
        names.push(dump_file.parse().unwrap());
    }

    format!("logging/{}{}.dat", dump_file.to_owned(), formatted_time)
}

#[cfg(test)]
mod tests {

    use super::*;
    use futures::stream;
    use futures::StreamExt;

    use tokio::sync::Mutex;

    #[cfg(feature = "logging")]
    use std::time::Duration;

    static MESSAGE_LOCK: Mutex<()> = Mutex::const_new(());
    #[tokio::test]
    async fn start_not_at_front() {
        let input: Vec<u8> = vec![0, 1, START_BYTE, END_BYTE];
        let input2: Vec<u8> = vec![END_BYTE, 1, START_BYTE, 3, END_BYTE, 5];
        let mut buffer: Vec<u8> = Vec::with_capacity(512);

        let _lock = MESSAGE_LOCK.lock().await;
        assert_eq!(
            stream::iter(
                get_messages(
                    &mut buffer,
                    &mut &*input,
                    #[cfg(feature = "logging")]
                    "test.dat"
                )
                .await
            )
            .collect::<Vec<Vec<u8>>>()
            .await,
            vec![vec![]]
        );

        assert_eq!(
            stream::iter(
                get_messages(
                    &mut buffer,
                    &mut &*input2,
                    #[cfg(feature = "logging")]
                    "test.dat"
                )
                .await
            )
            .collect::<Vec<Vec<u8>>>()
            .await,
            vec![vec![3]]
        );
    }

    #[tokio::test]
    #[cfg(feature = "logging")]
    async fn input_is_logged() {
        let input: Vec<u8> = vec![START_BYTE, 0, 1, END_BYTE];
        let input2: Vec<u8> = vec![START_BYTE, 3, 5, END_BYTE];
        let mut buffer: Vec<u8> = Vec::with_capacity(512);

        let dump_file = "test_log";

        let _lock = MESSAGE_LOCK.lock().await;
        {
            get_messages(&mut buffer, &mut &*input, dump_file).await;
            get_messages(&mut buffer, &mut &*input2, dump_file).await;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        assert_eq!(
            std::fs::read(fmt_filename_time(dump_file).await).unwrap(),
            vec![START_BYTE, 0, 1, END_BYTE, START_BYTE, 3, 5, END_BYTE]
        );
    }
}
