use itertools::Itertools;
use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    config::{gate::Config, ColorProfile, Side},
    missions::{
        action::{ActionConcurrentSplit, ActionDataConditional},
        basic::descend_depth_and_go_forward,
        extra::{AlwaysFalse, AlwaysTrue, Terminal},
        movement::{
            AdjustType, ClampX, FlipX, InvertX, ReplaceX, SetSideBlue, SetSideRed, SetX, SetY,
        },
        vision::{MidPoint, OffsetClass},
    },
    vision::{
        gate_cv::GateCV,
        gate_poles::{GatePoles, Target},
        nn_cv2::{OnnxModel, YoloClass},
        Offset2D,
    },
};

use super::{
    action::{
        wrap_action, ActionChain, ActionConcurrent, ActionExec, ActionMod, ActionSequence,
        ActionWhile, FirstValid, TupleSecond,
    },
    action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard},
    basic::{descend_and_go_forward, DelayAction},
    comms::StartBno055,
    extra::{CountFalse, CountTrue, OutputType},
    movement::{
        AdjustMovementAngle, LinearYawFromX, OffsetToPose, Stability2Adjust, Stability2Movement,
        Stability2Pos, ZeroMovement,
    },
    vision::{DetectTarget, ExtractPosition, VisionNorm, VisionNormOffset},
};

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

    const TOLERANCE: f32 = 0.3;

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

        let leftPole = detections.iter().filter(|d| *d.class()).collect_vec();
        let leftPole_avg_x = leftPole
            .iter()
            .map(|d| *d.position().x() as f32)
            .sum::<f32>();

        let rightPole = detections.iter().filter(|d| !*d.class()).collect_vec();
        let rightPole_avg_x = rightPole
            .iter()
            .map(|d| *d.position().x() as f32)
            .sum::<f32>();

        match gate_state {
            GateState::Align => match config.side {
                Side::Left => {
                    if leftPole.len() > 0 {
                        false_count = 0;
                        let mut correction;
                        if leftPole_avg_x < 0.2 {
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
                            correction = dbg!(config.correction_factor * leftPole_avg_x);
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
                    if rightPole.len() > 0 {
                        false_count = 0;
                        let mut correction;
                        if rightPole_avg_x < 0.2 {
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
                            correction = dbg!(config.correction_factor * rightPole_avg_x);
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

                yaw_target = (yaw_target
                    + (if let Side::Left = config.side {
                        config.yaw_adjustment
                    } else {
                        -config.yaw_adjustment
                    }));

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

        let rightPole = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::RightPole))
            .collect_vec();

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
                    let avg_x = (sawfish
                        .iter()
                        .map(|d| *d.position().x() as f32)
                        .sum::<f32>()
                        / sawfish.len() as f32);

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

pub fn gate_run_naive<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    let depth: f32 = -1.5;

    ActionSequence::new(
        ActionConcurrent::new(descend_and_go_forward(context), StartBno055::new(context)),
        ActionSequence::new(
            ActionWhile::new(ActionChain::new(
                VisionNormOffset::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::default(),
                ),
                TupleSecond::new(ActionConcurrent::new(
                    AdjustMovementAngle::new(context, depth),
                    CountTrue::new(3),
                )),
            )),
            ActionWhile::new(ActionChain::new(
                VisionNormOffset::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::default(),
                ),
                TupleSecond::new(ActionConcurrent::new(
                    AdjustMovementAngle::new(context, depth),
                    CountFalse::new(10),
                )),
            )),
        ),
    )
}

pub fn gate_run_complex<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<anyhow::Result<()>> + '_ {
    let depth: f32 = -1.40;

    act_nest!(
        ActionSequence::new,
        DelayAction::new(3.0),
        ActionConcurrent::new(
            descend_depth_and_go_forward(context, depth),
            StartBno055::new(context),
        ),
        act_nest!(
            ActionSequence::new,
            adjust_logic(context, depth, CountTrue::new(4)),
            adjust_logic(context, depth, CountFalse::new(4)),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            DelayAction::new(3.0),
            ZeroMovement::new(context, depth),
        ),
    )
}

pub fn gate_run_coinflip<
    'a,
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &'a Con,
    config: &Config,
) -> impl ActionExec<anyhow::Result<()>> + 'a {
    let depth = config.depth;

    act_nest!(
        ActionSequence::new,
        ActionConcurrent::new(
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            StartBno055::new(context),
        ),
        act_nest!(
            ActionSequence::new,
            adjust_logic(context, depth, CountTrue::new(config.true_count)),
            // adjust_logic(context, depth, CountFalse::new(10)),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 1.0, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            ActionWhile::new(act_nest!(
                ActionChain::new,
                VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default()),
                act_nest!(
                    wrap_action(ActionConcurrent::new, FirstValid::new),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Middle),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
                ),
                CountFalse::new(config.false_count),
            )),
            ActionChain::new(
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 0.5, 0.0, 0.0, None, depth),
                ),
                OutputType::<()>::default()
            ),
            DelayAction::new(0.0),
            ZeroMovement::new(context, depth),
        ),
    )
}

pub fn adjust_logic<
    'a,
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
    X: 'a + ActionMod<bool> + ActionExec<anyhow::Result<()>>,
>(
    context: &'a Con,
    depth: f32,
    end_condition: X,
) -> impl ActionExec<()> + 'a {
    const GATE_TRAVERSAL_SPEED: f32 = 0.2;

    ActionWhile::new(ActionChain::new(
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default()),
        ActionChain::new(
            TupleSecond::new(ActionConcurrent::new(
                ActionDataConditional::new(
                    //act_nest!(
                    //wrap_action(ActionConcurrent::new, FirstValid::new),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                    //DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(
                    //Target::Middle
                    //),
                    //),
                    ActionSequence::new(SetSideBlue::new(), Terminal::new()),
                    ActionDataConditional::new(
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                        ActionSequence::new(SetSideRed::new(), Terminal::new()),
                        Terminal::new(),
                    ),
                ),
                ActionDataConditional::new(
                    act_nest!(
                        wrap_action(ActionConcurrent::new, FirstValid::new),
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(
                            Target::Middle
                        ),
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                    ),
                    act_nest!(
                        ActionConcurrent::new,
                        act_nest!(
                            ActionChain::new,
                            OffsetClass::new(Target::Middle, Offset2D::<f64>::new(-0.05, 0.0)),
                            //OffsetClass::new(Target::Blue, Offset2D::<f64>::new(-0.1, 0.0)),
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::default(),
                            LinearYawFromX::<Stability2Adjust>::new(5.0),
                            ClampX::new(0.2),
                            SetY::<Stability2Adjust>::new(AdjustType::Adjust(0.02)),
                            FlipX::default(),
                        ),
                        AlwaysTrue::new(),
                    ),
                    ActionDataConditional::new(
                        DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
                        act_nest!(
                            ActionConcurrent::new,
                            act_nest!(
                                ActionChain::new,
                                ExtractPosition::new(),
                                MidPoint::new(),
                                OffsetToPose::default(),
                                InvertX::new(),
                                LinearYawFromX::<Stability2Adjust>::new(-7.0),
                                //ClampX::new(0.8),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(0.2)),
                                ReplaceX::new(),
                            ),
                            AlwaysTrue::new(),
                        ),
                        ActionConcurrent::new(
                            act_nest!(
                                ActionSequence::new,
                                Terminal::new(),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(0.4)),
                                SetX::<Stability2Adjust>::new(AdjustType::Replace(0.0)),
                            ),
                            AlwaysFalse::new(),
                        ),
                    ),
                ),
            )),
            TupleSecond::new(ActionConcurrentSplit::new(
                act_nest!(
                    ActionChain::new,
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, GATE_TRAVERSAL_SPEED, 0.0, 0.0, None, depth),
                    ),
                    OutputType::<()>::new(),
                ),
                end_condition,
            )),
        ),
    ))
}

pub fn gate_run_testing<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    let depth: f32 = -1.0;
    adjust_logic(context, depth, CountTrue::new(3))
}

enum GateState {
    Align,
    Approach,
}
