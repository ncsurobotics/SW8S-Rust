use std::{
    fs::{create_dir, File},
    sync::{LazyLock, Mutex},
};

use chrono::Local;

pub static LOGFILE: LazyLock<Mutex<File>> = LazyLock::new(|| {
    let _ = create_dir("console");
    Mutex::new(
        File::create(
            &("console/".to_string()
                + &Local::now().format("%Y-%m-%d_%H:%M:%S").to_string()
                + ".txt"),
        )
        .unwrap(),
    )
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

pub mod comms;
pub mod missions;
pub mod video_source;
pub mod vision;
