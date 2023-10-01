use std::{str::from_utf8, sync::Arc, time::Duration};

use sw8s_rust_lib::comms::meb::response::Statuses;
use tokio::{fs::File, sync::RwLock, time::timeout};

#[tokio::test]
async fn real_comms_read_no_error() {
    let mut err_msgs = Vec::new();

    Statuses::update_status(
        &mut Vec::with_capacity(512),
        &mut File::open("tests/comms/meb/meb_in.dat").await.unwrap(),
        &RwLock::default(),
        &RwLock::default(),
        &RwLock::default(),
        &Arc::default(),
        &RwLock::default(),
        &RwLock::default(),
        &mut err_msgs,
    )
    .await;

    print!("{}", from_utf8(&err_msgs).unwrap());
    assert!(err_msgs.is_empty());
}
