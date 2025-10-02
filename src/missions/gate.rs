use itertools::Itertools;
use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

use crate::{
    config::{gate::Config, ColorProfile, Side},
    vision::{
        gate_cv::GateCV,
        gate_poles::{GatePoles, Target},
        nn_cv2::OnnxModel,
    },
};

use super::{
    action::ActionExec,
    action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard},
    basic::DelayAction,
    vision::VisionNorm,
};

pub async fn gate_run_dead_reckon<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting Procedural Gate");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    // let _ = cb
    //     .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
    //     .await;

    // sleep(Duration::from_secs_f32(config.init_duration)).await;
    // let mut mult = 1.0;
    // if let Side::Left = config.side {
    //     mult = -1.0;
    // } else {
    //     mult = 1.0;
    // }
    // let _ = cb
    //     .stability_2_speed_set(
    //         config.strafe_speed * mult,
    //         0.0,
    //         0.0,
    //         0.0,
    //         initial_yaw,
    //         config.depth,
    //     )
    //     .await;

    // sleep(Duration::from_secs_f32(config.strafe_duration)).await;

    let _ = cb
        .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    sleep(Duration::from_secs_f32(config.traversal_duration)).await;

    let _ = cb
        .stability_1_speed_set(0.0, 0.0, 0.0, 0.0, 0.0, config.depth)
        .await;
}

pub async fn gate_run_cv_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
    color_profile: &ColorProfile,
) {
    #[cfg(feature = "logging")]
    logln!("Starting Procedural Gate");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    // let mut vision = VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GateCV::default());
    let mut vision =
        VisionNorm::<Con, GateCV, f64>::new(context, GateCV::from_color_profile(color_profile));

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

    // const TOLERANCE: f32 = 0.3;

    let mut gate_state = GateState::Align;
    let mut yaw_target = 0.0;
    let mut true_count = 0;
    let mut false_count = 0;

    loop {
        #[allow(unused_variables)]
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        let left_pole = detections.iter().filter(|d| *d.class()).collect_vec();
        let left_pole_avg_x = left_pole
            .iter()
            .map(|d| *d.position().x() as f32)
            .sum::<f32>();

        let right_pole = detections.iter().filter(|d| !*d.class()).collect_vec();
        let right_pole_avg_x = right_pole
            .iter()
            .map(|d| *d.position().x() as f32)
            .sum::<f32>();

        match gate_state {
            GateState::Align => match config.side {
                Side::Left => {
                    if left_pole.len() > 0 {
                        false_count = 0;
                        let correction;
                        if left_pole_avg_x < 0.2 {
                            true_count += 1;
                            if true_count >= config.true_count {
                                #[cfg(feature = "logging")]
                                logln!("ALIGNED");
                                if let Some(current_angle) = cb.responses().get_angles().await {
                                    let current_yaw = *current_angle.yaw();
                                    yaw_target = current_yaw;
                                }
                                gate_state = GateState::Approach;
                            } else {
                                #[cfg(feature = "logging")]
                                logln!("true_count: {true_count}/4");
                            }
                        } else {
                            correction = dbg!(config.correction_factor * left_pole_avg_x);
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
                                -config.yaw_speed,
                                0.0,
                                0.0,
                                config.depth,
                            )
                            .await;

                        false_count += 1;
                        #[cfg(feature = "logging")]
                        logln!("NO DETECTIONS");
                        if false_count >= 100 {
                            #[cfg(feature = "logging")]
                            logln!("KILLED NO DET");
                            break;
                        }
                    }
                }

                Side::Right => {
                    if right_pole.len() > 0 {
                        false_count = 0;
                        let correction;
                        if right_pole_avg_x < 0.2 {
                            true_count += 1;
                            if true_count >= config.true_count {
                                #[cfg(feature = "logging")]
                                logln!("ALIGNED");
                                if let Some(current_angle) = cb.responses().get_angles().await {
                                    let current_yaw = *current_angle.yaw();
                                    yaw_target = current_yaw;
                                }
                                gate_state = GateState::Approach;
                            } else {
                                #[cfg(feature = "logging")]
                                logln!("true_count: {true_count}/4");
                            }
                        } else {
                            correction = dbg!(config.correction_factor * right_pole_avg_x);
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
                                config.yaw_speed,
                                0.0,
                                0.0,
                                config.depth,
                            )
                            .await;

                        false_count += 1;
                        #[cfg(feature = "logging")]
                        logln!("NO DETECTIONS");
                        if false_count >= 100 {
                            #[cfg(feature = "logging")]
                            logln!("KILLED NO DET");
                            break;
                        }
                    }
                }
            },
            GateState::Approach => {
                #[cfg(feature = "logging")]
                logln!("APPROACH");

                let strafe_direction = if let Side::Left = config.side {
                    -1.0
                } else {
                    1.0
                };

                let _ = cb
                    .stability_2_speed_set(
                        config.strafe_speed * strafe_direction,
                        0.0,
                        0.0,
                        0.0,
                        yaw_target,
                        config.depth,
                    )
                    .await;

                sleep(Duration::from_secs(config.strafe_duration as u64)).await;

                yaw_target = yaw_target
                    + (if let Side::Left = config.side {
                        config.yaw_adjustment
                    } else {
                        -config.yaw_adjustment
                    });

                let _ = cb
                    .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, yaw_target, config.depth)
                    .await;

                sleep(Duration::from_secs(config.init_duration as u64)).await;

                let _ = cb
                    .stability_2_speed_set(0.0, config.speed, 0.0, 0.0, yaw_target, config.depth)
                    .await;

                sleep(Duration::from_secs(config.traversal_duration as u64)).await;

                break;
            }
        }
    }
}

pub async fn gate_run_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting Procedural Gate");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let mut vision =
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default());

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

    const TOLERANCE: f32 = 0.3;

    let mut true_count = 0;

    loop {
        #[allow(unused_variables)]
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        // let right_pole = detections
        //     .iter()
        //     .filter(|d| matches!(d.class().identifier, Target::RightPole))
        //     .collect_vec();

        /* let middle = detections
        .iter()
        .filter(|d| matches!(d.class().identifier, Target::Middle))
        .collect_vec(); */

        let shark = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Sawfish))
            .collect_vec();

        let sawfish = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Shark))
            .collect_vec();

        let mut traversal_timer = DelayAction::new(8.0); // forward duration in second

        match config.side {
            Side::Left => {
                if !shark.is_empty() {
                    // Center on average x of blue
                    let avg_x = shark.iter().map(|d| *d.position().x() as f32).sum::<f32>()
                        / shark.len() as f32;

                    #[cfg(feature = "logging")]
                    logln!("SHARK AVG X: {}", avg_x);

                    if avg_x.abs() > TOLERANCE {
                        let correction = 0.4 * avg_x;
                        let fwd = 0.0;

                        let _ = cb
                            .stability_2_speed_set(
                                correction,
                                fwd,
                                0.0,
                                0.0,
                                initial_yaw,
                                config.depth,
                            )
                            .await;
                    } else {
                        let fwd = config.speed;
                        let correction = 0.05;
                        true_count += 1;

                        if true_count >= config.true_count {
                            let _ = cb
                                .stability_2_speed_set(
                                    correction,
                                    fwd,
                                    0.0,
                                    0.0,
                                    initial_yaw,
                                    config.depth,
                                )
                                .await;
                            // let _ = cb
                            //     .stability_1_speed_set(correction, fwd, 0.0, 0.0, 0.0, config.depth)
                            //     .await;

                            traversal_timer.execute().await;
                            break;
                        }
                    }
                } else {
                    // Fallback search behavior
                    #[cfg(feature = "logging")]
                    logln!("LEFT: Missing Features, Fallback");

                    let correction = -0.2;
                    let fwd = 0.05;

                    let _ = cb
                        .stability_2_speed_set(correction, fwd, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                    // let _ = cb
                    // .stability_1_speed_set(correction, fwd, 0.0, 0.0, 0.0, config.depth)
                    // .await;

                    DelayAction::new(1.0).execute().await;
                }
            }

            Side::Right => {
                if !sawfish.is_empty() {
                    // Center on average x of blue
                    let avg_x = sawfish
                        .iter()
                        .map(|d| *d.position().x() as f32)
                        .sum::<f32>()
                        / sawfish.len() as f32;

                    #[cfg(feature = "logging")]
                    logln!("SAWFISH AVG X: {}", avg_x);

                    if avg_x.abs() > TOLERANCE {
                        let correction = 0.4 * avg_x;
                        let fwd = 0.05;

                        let _ = cb
                            .stability_2_speed_set(
                                correction,
                                fwd,
                                0.0,
                                0.0,
                                initial_yaw,
                                config.depth,
                            )
                            .await;
                        // let _ = cb
                        //     .stability_1_speed_set(correction, fwd, 0.0, 0.0, 0.0, config.depth)
                        //     .await;
                    } else {
                        let fwd = config.speed;
                        let correction = 0.05;
                        true_count += 1;

                        if true_count >= config.true_count {
                            let _ = cb
                                .stability_2_speed_set(
                                    correction,
                                    fwd,
                                    0.0,
                                    0.0,
                                    initial_yaw,
                                    config.depth,
                                )
                                .await;
                            // let _ = cb
                            // .stability_1_speed_set(correction, fwd, 0.0, 0.0, 0.0, config.depth)
                            // .await;

                            traversal_timer.execute().await;
                            break;
                        }
                    }
                } else {
                    // Fallback search behavior
                    #[cfg(feature = "logging")]
                    logln!("RIGHT: Missing Features, Fallback");

                    let correction = 0.2;
                    let fwd = 0.05;

                    let _ = cb
                        .stability_2_speed_set(correction, fwd, 0.0, 0.0, initial_yaw, config.depth)
                        .await;
                    // let _ = cb
                    //     .stability_1_speed_set(correction, fwd, 0.0, 0.0, 0.0, config.depth)
                    //     .await;
                }
            }
        }
    }
}

enum GateState {
    Align,
    Approach,
}
