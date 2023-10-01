use std::{str::from_utf8, sync::Arc};

use sw8s_rust_lib::comms::{auv_control_board::response::find_end, meb::response::Statuses};
use tokio::sync::RwLock;

#[tokio::test]
async fn real_comms_read_no_error() {
    let mut buffer = Vec::with_capacity(512);
    let mut bytes: Vec<u8> = include_bytes!("meb_in.dat").to_vec();
    let mut errors: Vec<(u8, Vec<u8>, Vec<u8>)> = Vec::new();
    let mut prev_byte = 254;
    let mut total_chunks = 0;

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
            &RwLock::default(),
            &RwLock::default(),
            &mut err_msg,
        )
        .await;

        if !err_msg.is_empty() {
            errors.push((prev_byte, byte_chunk.clone(), err_msg));
        }
        prev_byte = *byte_chunk.last().unwrap_or(&0);
    }

    errors.clone().into_iter().for_each(|entry| {
        println!("Prev byte: {}", entry.0);
        println!("Chunk: {:?}", entry.1);
        print!("{}", from_utf8(&entry.2).unwrap());
    });
    println!(
        "\n{} errors in {} entries, {}% error",
        errors.len(),
        total_chunks,
        ((errors.len() as f32) / (total_chunks as f32)) * 100.0
    );

    assert!(errors.is_empty());
}
