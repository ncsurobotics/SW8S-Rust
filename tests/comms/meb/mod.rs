use std::{str::from_utf8, sync::Arc};

use sw8s_rust_lib::comms::{auv_control_board::response::find_end, meb::response::Statuses};
use tokio::sync::{Mutex, RwLock};

#[tokio::test]
async fn real_comms_read_no_error() {
    let mut buffer = Vec::with_capacity(512);
    let mut bytes: Vec<u8> = include_bytes!("meb_in.dat").to_vec();
    let mut prev_byte = 254;
    let mut errors: usize = 0;
    let mut total_chunks: usize = 0;

    while let Some((end_idx, _)) = find_end(&bytes) {
        total_chunks += 1;
        let mut err_msg = Vec::new();
        let byte_chunk: Vec<u8> = bytes.drain(0..=end_idx).collect();

        Statuses::update_status(
            &mut buffer,
            &mut &*byte_chunk,
            &RwLock::default(),
            &RwLock::default(),
            &RwLock::default(),
            &Arc::default(),
            &Arc::new(Mutex::new(vec![false; 24])),
            &RwLock::default(),
            &RwLock::default(),
            &mut err_msg,
        )
        .await;

        if !err_msg.is_empty() {
            errors += 1;
            println!("Prev byte: {}", prev_byte);
            println!("Chunk: {:?}", byte_chunk);
            println!("{}", from_utf8(&err_msg).unwrap());
        }
        prev_byte = *byte_chunk.last().unwrap_or(&0);
    }

    let percent_error = ((errors as f32) / (total_chunks as f32)) * 100.0;

    println!(
        "\n{} errors in {} entries, {}% error",
        errors, total_chunks, percent_error
    );

    assert!(percent_error < 1.0);
}
