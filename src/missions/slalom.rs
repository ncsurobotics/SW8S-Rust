use itertools::Itertools;
use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

use bluerobotics_ping::{
    common::{DeviceInformationStruct, ProtocolVersionStruct},
    device::{Ping360, PingDevice},
    ping360::AutoDeviceDataStruct,
};

use tokio::select;
use tokio_serial::{SerialPort, SerialPortBuilderExt};
use tokio_util::sync::CancellationToken;

use crate::config::sonar::Config as SonarConfig;
use std::f64::consts::PI;

use geo::{polygon, ConvexHull, LineString};
use hdbscan::{Center, Hdbscan};

use super::action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::{
    config::slalom::Config,
    config::slalom::Side::*,
    missions::{action::ActionExec, vision::VisionNorm},
    vision::{nn_cv2::OnnxModel, slalom::Slalom, slalom::Target},
};
pub async fn slalom_sonar<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    slalom_config: &Config,
    cfg: &SonarConfig,
    cancel: CancellationToken,
) {
    const INTESNTIY_THRESH: u8 = 100;
    const MAX_DISTANCE: f64 = 20.0;
    const SPEED_OF_SOUND: f64 = 1500.0; //m/s

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    #[cfg(feature = "logging")]
    logln!("Initializing sonar with: {:?}", cfg.serial_port);
    let port = loop {
        match tokio_serial::new(cfg.serial_port.to_string_lossy(), cfg.serial_baud_rate)
            .open_native_async()
        {
            Ok(port) => break port,
            Err(e) => {
                #[cfg(feature = "logging")]
                logln!("Error opening serial port: {}", e);
            }
        }
    };

    // let mut port_clone = port.try_clone();

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
    loop {
        if let Err(e) = ping360.motor_off().await {
            #[cfg(feature = "logging")]
            logln!("Failed to reset sonar unit: {e:#?}");
        } else {
            break;
        }
    }

    let (protocol_version, device_information) =
        tokio::try_join!(ping360.protocol_version(), ping360.device_information())
            .expect("Failed to get device data!");

    // let _ = cb
    // .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, -1.25)
    // .await;

    #[cfg(feature = "logging")]
    logln!("Starting sonar auto transmit");
    let at = cfg.auto_transmit;
    loop {
        if let Err(e) = ping360
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
        } else {
            break;
        }
    }

    let mut data: Vec<AutoDeviceDataStruct> = Vec::new();

    #[cfg(feature = "logging")]
    logln!("Recording data");
    loop {
        select! {
            _ = cancel.cancelled() => { break; },
            r = ping360.auto_device_data() => {
                if let Ok(d) = r {
                    if (d.angle == 180) {
                        break;
                    }
                    data.push(d);
                    #[cfg(feature = "logging")]
                    logln!("Got data");
                }
            }
        }
    }
    // let _ = port_clone.expect("NO CLONE").set_break();

    let mut points = Vec::new();
    let mut points_f32 = Vec::new();

    for packet in data {
        let angle_rad: f64 = ((packet.angle as f64) * (PI / 200.0)).into();
        #[cfg(feature = "logging")]
        logln!("Checking Angle {}", &angle_rad * 57.29577951308);
        let sample_period = (packet.sample_period as f64) * 25e-9;
        let num_samples = packet.number_of_samples as usize;

        for (i, &intensity) in packet.data.iter().enumerate().take(num_samples) {
            if intensity < INTESNTIY_THRESH {
                continue;
            }

            let range = (i as f64) * sample_period * SPEED_OF_SOUND / 2.0;
            if range > MAX_DISTANCE || range < 0.75 {
                continue;
            }

            let x = range * angle_rad.cos();
            let y = range * angle_rad.sin();

            let center_vec = vec![x, y];
            points.push(center_vec);
            points_f32.push(vec![x as f32, y as f32]);
        }
    }

    #[cfg(feature = "logging")]
    logln!("NUM POINTS: {}", points.len());

    if points.len() < 5 {
        #[cfg(feature = "logging")]
        logln!("Not enough points for clustering!");
        return;
    }

    let clusterer = Hdbscan::default_hyper_params(&points);
    let labels = clusterer.cluster().unwrap();
    let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();

    // let mut label_map: std::collections::HashMap<i32, Vec<[f64; 2]>> =
    //     std::collections::HashMap::new();
    // for (i, &label) in labels.iter().enumerate() {
    //     if label >= 0 {
    //         label_map.entry(label).or_default().push(points[i]);
    //     }
    // }

    for center in centroids {
        let cluster_x = center[0];
        let cluster_y = center[1];
        let cluster_angle = cluster_y.atan2(cluster_x);

        #[cfg(feature = "logging")]
        logln!(
            "X: {}, Y: {}, A: {}",
            cluster_x,
            cluster_y,
            cluster_angle * 57.29577951
        );
    }
}

pub async fn slalom<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting slalom");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let mut vision = VisionNorm::<Con, Slalom<OnnxModel>, f64>::new(context, Slalom::default());

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let mut start_detections = 0;
    let mut end_detections = 0;

    // Begin stationary, we will only start forward if we detect nothing
    let _ = cb
        .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    loop {
        #[cfg(feature = "logging")]
        logln!("ATTEMPTING TO DO SLALOM DETECTION PLEASE");
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        if detections.len() == 0 {
            #[cfg(feature = "logging")]
            logln!("NO DETECTIONS");

            // We see nothing, meaning we have either not reached slalom yet, passed through it already, or are misaligned.
            // If we have not reached it yet, we need to go forward
            // If we have passed through already, end the mission
            if start_detections > config.start_detections {
                if end_detections > config.end_detections {
                    break;
                } else {
                    end_detections += 1;
                    #[cfg(feature = "logging")]
                    logln!("End count: {}/{}", end_detections, config.end_detections);
                }
            } else {
                #[cfg(feature = "logging")]
                logln!(
                    "Resetting start counter from {}/{}",
                    start_detections,
                    config.start_detections
                );
                start_detections = 0;
                let _ = cb
                    .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
                    .await;
            }
            continue;
        } else {
            if start_detections < config.start_detections {
                start_detections += 1;
                #[cfg(feature = "logging")]
                logln!(
                    "Start count: {}/{}",
                    start_detections,
                    config.start_detections
                );
            } else {
                #[cfg(feature = "logging")]
                logln!(
                    "Resetting end counter from {}/{}",
                    end_detections,
                    config.end_detections
                );
                end_detections = 0;
            }
        }

        let middle = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Middle))
            .collect_vec();

        let side = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Side))
            .collect_vec();

        if middle.len() > 0 {
            // Use the average x coord of all middle detections as the middle of slalom
            let middle_x =
                middle.iter().map(|d| *d.position().x() as f32).sum::<f32>() / middle.len() as f32;

            if side.len() > 0 {
                #[cfg(feature = "logging")]
                logln!("Got middle and side poles");
                // We have the middle and at least one side pole
                // Split the side pole detections into left and right groups based on their x coords
                let mut sum_l = 0.0;
                let mut sum_r = 0.0;
                let mut count_l = 0.0;
                let mut count_r = 0.0;

                for x in side.iter().map(|d| *d.position().x() as f32) {
                    // If the side poles x coord is greater than the middle pole, it's on the left
                    // Haven't checked if this is actually the right side, it might be flipped
                    if x > middle_x {
                        sum_l += x;
                        count_l += 1.0;
                    } else if x < middle_x {
                        sum_r += x;
                        count_r += 1.0;
                    }
                }

                let (sum, count) = match config.side {
                    Left => (sum_l, count_l),
                    Right => (sum_r, count_r),
                };

                // Find average x coords for l/r group of side poles
                let side_x = if count > 0.0 { sum / count } else { 0.0 };

                // Set target x to the average of the chosen side pole and middle pole's x coords
                let x = (side_x + middle_x) / 2.0;

                // Strafe to the left right towards the midpoint between the middle pole and the chosen side pole
                let _ = cb
                    .stability_2_speed_set(
                        x as f32,
                        config.speed,
                        0.0,
                        0.0,
                        initial_yaw,
                        config.depth,
                    )
                    .await;
            } else {
                // We have no side pole, so just align to the middle pole
                // Once the middle poles x is below a certain value, do a dumb translation in the correct direction
                if middle_x <= config.centered_threshold {
                    #[cfg(feature = "logging")]
                    logln!("Got only middle pole, finished centering");
                    let x = match config.side {
                        Left => config.speed,
                        Right => config.speed * -1.0,
                    };

                    let _ = cb
                        .stability_2_speed_set(x, config.speed, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                    sleep(Duration::from_secs(config.dumb_strafe_secs)).await;
                    let _ = cb
                        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                } else {
                    #[cfg(feature = "logging")]
                    logln!("Got only middle pole, centering");
                    let _ = cb
                        .stability_2_speed_set(middle_x, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                }
            }
        } else {
            if side.len() > 0 {
                #[cfg(feature = "logging")]
                logln!("Got only side poles");
                // We see at least a side pole, but no middle
                // This means we are probably seeing a side pole, so translate towards it
                let x =
                    side.iter().map(|d| *d.position().x() as f32).sum::<f32>() / side.len() as f32;
                let _ = cb
                    .stability_2_speed_set(x, 0.0, 0.0, 0.0, initial_yaw, config.depth)
                    .await;
            }
        }
    }

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    #[cfg(feature = "logging")]
    logln!("Finished slalom");
}
