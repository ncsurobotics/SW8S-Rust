use bluerobotics_ping::{
    common::{DeviceInformationStruct, ProtocolVersionStruct},
    device::{Ping360, PingDevice},
    ping360::AutoDeviceDataStruct,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::BufWriter,
    path::PathBuf,
    time::SystemTime,
};
use tokio::{io::WriteHalf, select};
use tokio_serial::{SerialPort, SerialPortBuilderExt, SerialStream};
use tokio_util::sync::CancellationToken;

use super::action_context::{GetControlBoard, GetMainElectronicsBoard};
use crate::config::sonar::Config;

pub async fn sonar<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard,
>(
    context: &Con,
    cfg: &Config,
    cancel: CancellationToken,
) {
    #[cfg(feature = "logging")]
    logln!("Starting sonar");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    #[cfg(feature = "logging")]
    logln!("Initializing sonar with: {:?}", cfg.serial_port);
    let port = loop {
        match tokio_serial::new(cfg.serial_port.to_string_lossy(), cfg.serial_baud_rate)
            .open_native_async()
        {
            Ok(port) => break port,
            #[allow(unused_variables)]
            Err(e) => {
                #[cfg(feature = "logging")]
                logln!("Error opening serial port: {}", e);
            }
        }
    };

    #[allow(unused_variables)]
    port.clear(tokio_serial::ClearBuffer::All)
        .unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Failed to clear sonar serial port: {}", e);
        });

    let ping360 = Ping360::new(port);

    // #[cfg(feature = "logging")]
    // logln!("Reseting sonar unit");
    // loop {
    //     if let Err(e) = ping360.reset(cfg.bootloader as u8, 0).await {
    //         #[cfg(feature = "logging")]
    //         logln!("Failed to reset sonar unit: {e:#?}");
    //     } else {
    //         break;
    //     }
    // }

    #[cfg(feature = "logging")]
    logln!("Reseting MOTOR sonar unit");
    #[allow(unused_variables)]
    while let Err(e) = ping360.motor_off().await {
        #[cfg(feature = "logging")]
        logln!("Failed to reset sonar unit: {e:#?}");
    }

    let (protocol_version, device_information) =
        tokio::try_join!(ping360.protocol_version(), ping360.device_information())
            .expect("Failed to get device data!");

    // let _ = cb
    //     .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, -1.25)
    //     .await;

    #[cfg(feature = "logging")]
    logln!("Opening log file");
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let directory = "./logging/sonar/";
    let filename = format!("{time}.log");
    let path = PathBuf::from(directory).join(filename);
    let mut open_options = OpenOptions::new();
    open_options.append(true).create(true);

    let file = open_options
        .open(path.as_path())
        .unwrap_or_else(|e| match path.parent() {
            Some(parent) => {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create log file parent directory: {e}"))
                    .unwrap();
                open_options
                    .open(path)
                    .map_err(|e| format!("Failed to open log file: {e}"))
                    .unwrap()
            }
            None => {
                panic!("Failed to open log file: {e}");
            }
        });

    let file = BufWriter::new(file);

    #[cfg(feature = "logging")]
    logln!("Starting sonar auto transmit");
    let at = cfg.auto_transmit;
    #[allow(unused_variables)]
    while let Err(e) = ping360
        .auto_transmit(
            at.mode,
            at.gain_setting as u8,
            at.transmit_duration,
            at.sample_period,
            at.transmit_frequency,
            at.number_of_samples,
            at.start_angle,
            at.stop_angle,
            at.num_steps,
            at.delay,
        )
        .await
    {
        #[cfg(feature = "logging")]
        logln!("Failed to start sonar auto transmit: {e:#?}");
    }

    let mut data: Vec<AutoDeviceDataStruct> = Vec::new();

    #[cfg(feature = "logging")]
    logln!("Recording data");
    loop {
        select! {
            _ = cancel.cancelled() => { break; },
            r = ping360.auto_device_data() => {
                if let Ok(d) = r {
                    let angle_clone = d.angle;
                    data.push(d);
                    #[cfg(feature = "logging")]
                    logln!("Got data {}", angle_clone);
                    if angle_clone >= at.stop_angle {
                        break;
                    }
                }
            }
        }
    }

    let log = SonarLogFile {
        protocol_version,
        device_information,
        data,
    };

    serde_json::to_writer_pretty(file, &log).expect("Failed to write sonar log file");
}

#[derive(Serialize, Deserialize)]
struct SonarLogFile {
    protocol_version: ProtocolVersionStruct,
    device_information: DeviceInformationStruct,
    data: Vec<AutoDeviceDataStruct>,
}
