use std::{
    fs::{create_dir, File},
    sync::{LazyLock, Mutex},
};

use chrono::Local;

pub static TIMESTAMP: LazyLock<String> =
    LazyLock::new(|| Local::now().format("%Y-%m-%d_%H:%M:%S").to_string());

pub static LOGFILE: LazyLock<Mutex<File>> = LazyLock::new(|| {
    let _ = create_dir("console");
    Mutex::new(File::create(&("console/".to_string() + &TIMESTAMP + ".txt")).unwrap())
});

#[macro_export]
macro_rules! logln {
    () => { {
            use std::io::Write;
        println!(); let _ = writeln!($crate::LOGFILE.lock().unwrap(), "");
    }};
    ($($arg:tt)*) => {
        {
            use std::io::Write;

            println!($($arg)*);
            let _ = writeln!($crate::LOGFILE.lock().unwrap(), $($arg)*);
        }
    };
}

/// Set to `1.0` or `-1.0`.
///
/// `1.0` is counterclockwise to find buoy, clockwise to find octagon.
pub const POOL_YAW_SIGN: f32 = -1.0;

pub mod comms;
pub mod config;
pub mod missions;
pub mod video_source;
pub mod vision;
