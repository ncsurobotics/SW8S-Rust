use hdbscan::{Center, Hdbscan};
use itertools::Itertools;
use std::f64::consts::PI;

use tokio::{
    io::WriteHalf,
    select,
    time::{sleep, Duration},
};
use tokio_serial::{SerialPort, SerialPortBuilderExt, SerialStream};
use tokio_util::sync::CancellationToken;

use bluerobotics_ping::{
    device::{Ping360, PingDevice},
    ping360::AutoDeviceDataStruct,
};

use super::action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::{
    config::{slalom::Config, sonar::Config as SonarConfig, ColorProfile, Side::*},
    missions::{
        action::ActionExec,
        basic::DelayAction,
        vision::{VisionNorm, VisionNormAngle},
    },
};

// TODO: Consider filtering detections by angle (poles will always be upright)
pub async fn slalom<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
    flip: bool,
    color_profile: &ColorProfile,
) {
    use crate::vision::slalom::Slalom;
    #[cfg(feature = "logging")]
    logln!("Starting slalom");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let mut vision = VisionNormAngle::<Con, Slalom, f64>::new(
        context,
        Slalom::from_color_profile(color_profile),
    );

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let mut yaw_target = 0.0;
    let mut true_count = 0;
    let mut false_count = 0;
    let mut init_timer = DelayAction::new(config.init_duration);
    let mut traversal_timer = DelayAction::new(config.traversal_duration); // forward duration in second
    let mut strafe_timer = DelayAction::new(config.strafe_duration);

    enum SlalomState {
        Align,
        Approach,
    }

    let mut slalom_state = SlalomState::Align;

    // let _ = cb
    //     .stability_2_speed_set(0.05, config.speed, 0.0, 0.0, initial_yaw, config.depth)
    //     .await;
    // init_timer.execute().await;

    #[cfg(feature = "logging")]
    logln!("Starting slalom detection");

    // Default left, right if flipped
    let _ = cb
        .stability_1_speed_set(
            0.0,
            0.0,
            if flip {
                config.yaw_speed
            } else {
                -config.yaw_speed
            },
            0.0,
            0.0,
            config.depth,
        )
        .await;
    'detections: loop {
        #[allow(unused_variables)]
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!(
                "Getting slalom detection resulted in error: `{e}`\n\tUsing empty detection vec"
            );
            vec![]
        });

        let mut positions = detections
            .into_iter()
            .filter_map(|d| d.class().then_some(d.position().clone()));

        match slalom_state {
            SlalomState::Align => {
                #[cfg(feature = "logging")]
                logln!("ALIGN");

                if let Some(position) = positions.next() {
                    let x = *position.x() as f32;
                    let mut correction = 0.0;
                    if x.abs() < 0.2 {
                        true_count += 1;
                        if true_count >= 4 {
                            correction = 0.0;
                            slalom_state = SlalomState::Approach;
                            if let Some(current_angle) = cb.responses().get_angles().await {
                                let current_yaw = *current_angle.yaw();
                                yaw_target = current_yaw;
                            }
                        } else {
                            #[cfg(feature = "logging")]
                            logln!("true_count: {true_count}/4");
                        }
                    } else {
                        correction = 0.5 * x;
                        let _ = cb
                            .stability_1_speed_set(0.0, 0.0, correction, 0.0, 0.0, config.depth)
                            .await;
                    }
                } else {
                    false_count += 1;
                    if false_count >= 100 {
                        break 'detections;
                    }
                }
            }

            SlalomState::Approach => {
                #[cfg(feature = "logging")]
                logln!("APPROACH");

                let _ = cb
                    .stability_2_speed_set(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        (yaw_target
                            + (if let Left = config.side {
                                -config.yaw_adjustment
                            } else {
                                config.yaw_adjustment
                            })) as f32,
                        config.depth,
                    )
                    .await;

                init_timer.execute().await;

                let _ = cb
                    .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, 0.0, config.depth)
                    .await;

                traversal_timer.execute().await;
                break 'detections;
            }
        }

        // The current implementation is guaranteed to return exactly 1 item
    }
}
