use tokio::{
    io::WriteHalf,
    time::{sleep, Duration},
};
use tokio_serial::SerialStream;

use super::action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::{
    config::{slalom::Config, ColorProfile, Side::*},
    missions::{action::ActionExec, vision::VisionNormAngle},
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
        Slalom::from_color_profile(color_profile, config.area_bounds.clone()),
    );

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    sleep(Duration::from_secs(3)).await;

    let mut yaw_target = 0.0;
    let mut true_count = 0;
    let mut false_count = 0;

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
                    false_count = 0;
                    let x = *position.x() as f32;
                    dbg!(&x);
                    let mut correction = 0.0;
                    let _ = correction;
                    if x.abs() < 0.2 {
                        true_count += 1;
                        if true_count >= 4 {
                            correction = 0.0;
                            let _ = correction;
                            #[cfg(feature = "logging")]
                            logln!("ALIGNED");
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
                        correction = dbg!(config.correction_factor * x);
                        let _ = cb
                            .stability_1_speed_set(0.0, 0.0, correction, 0.0, 0.0, config.depth)
                            .await;
                    }
                } else {
                    #[cfg(feature = "logging")]
                    logln!("SEARCHING");

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

                    false_count += 1;
                    #[cfg(feature = "logging")]
                    logln!("NO DETECTIONS");
                    if false_count >= 500 {
                        #[cfg(feature = "logging")]
                        logln!("KILLED NO DET");
                        break 'detections;
                    }
                }
            }

            SlalomState::Approach => {
                #[cfg(feature = "logging")]
                logln!("APPROACH");

                let strafe_direction = if let Left = config.side { -1.0 } else { 1.0 };

                let _ = cb
                    .stability_2_speed_set(
                        config.speed * strafe_direction,
                        0.0,
                        0.0,
                        0.0,
                        yaw_target,
                        config.depth,
                    )
                    .await;

                sleep(Duration::from_secs(config.strafe_duration as u64)).await;

                yaw_target = yaw_target
                    + (if let Left = config.side {
                        config.yaw_adjustment
                    } else {
                        -config.yaw_adjustment
                    });

                let _ = cb
                    .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, yaw_target, config.depth)
                    .await;

                sleep(Duration::from_secs(config.init_duration as u64)).await;

                // init_timer.execute().await;

                // let _ = cb
                //     .stability_1_speed_set(0.0, config.speed, 0.0, 0.0, 0.0, config.depth)
                //     .await;

                //
                let _ = cb
                    .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, yaw_target, config.depth)
                    .await;

                // traversal_timer.execute().await;
                sleep(Duration::from_secs(config.traversal_duration as u64)).await;

                break 'detections;
            }
        }

        // The current implementation is guaranteed to return exactly 1 item
    }
}
