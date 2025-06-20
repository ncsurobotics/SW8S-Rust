use std::f32::consts::PI;

use itertools::Itertools;
use serde::de::IntoDeserializer;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    config::gate::{Config, Side},
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

pub async fn gate_run_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting Procedural Gate");

    let cb = context.get_control_board();
    cb.bno055_periodic_read(true).await;

    let mut vision =
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default());

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;

    let mut current_yaw = initial_yaw;

    loop {
        if let Some(current_angle) = cb.responses().get_angles().await {
            current_yaw = *current_angle.yaw() as f32;
        }

        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        let pole = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Pole))
            .collect_vec();

        let middle = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Middle))
            .collect_vec();

        let red = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Red))
            .collect_vec();

        let blue = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Blue))
            .collect_vec();

        let mut traversal_started = false;
        let mut traversal_timer = DelayAction::new(9.5); // forward duration in seconds

        let mut true_count = 0;

        match config.side {
            Side::Left => {
                if blue.len() > 0 {
                    // Center on average x of blue
                    let avg_x = blue.iter().map(|d| *d.position().x() as f32).sum::<f32>()
                        / blue.len() as f32;

                    #[cfg(feature = "logging")]
                    logln!("AVG X: {}", avg_x);

                    #[cfg(feature = "logging")]
                    logln!("True Count: {}", true_count);

                    if avg_x.abs() > 0.1 {
                        let correction = -0.5 * avg_x;
                        let fwd = 0.0;
                        let x_speed = -fwd * f32::sin(current_yaw * (PI / 180.0))
                            + correction * f32::cos(current_yaw * (PI / 180.0));
                        let y_speed = fwd * f32::cos(current_yaw * (PI / 180.0))
                            + correction * f32::sin(current_yaw * (PI / 180.0));
                        let _ = cb
                            .stability_2_speed_set(
                                x_speed,
                                y_speed,
                                0.0,
                                0.0,
                                initial_yaw,
                                config.depth,
                            )
                            .await;
                    } else {
                        let fwd = config.speed;
                        let correction = 0.0;
                        let x_speed = -fwd * f32::sin(current_yaw * (PI / 180.0))
                            + correction * f32::cos(current_yaw * (PI / 180.0));
                        let y_speed = fwd * f32::cos(current_yaw * (PI / 180.0))
                            + correction * f32::sin(current_yaw * (PI / 180.0));

                        true_count += 1;

                        if true_count > config.true_count {
                            let _ = cb
                                .stability_2_speed_set(
                                    x_speed,
                                    y_speed,
                                    0.0,
                                    0.0,
                                    initial_yaw,
                                    config.depth,
                                )
                                .await;
                            traversal_timer.execute().await;
                            break;
                        }
                    }
                } else {
                    // Fallback search behavior
                    #[cfg(feature = "logging")]
                    logln!("LEFT: Missing Features, Fallback");

                    let correction = -0.2;
                    let fwd = 0.0;
                    let x_speed = -fwd * f32::sin(current_yaw * (PI / 180.0))
                        + correction * f32::cos(current_yaw * (PI / 180.0));
                    let y_speed = fwd * f32::cos(current_yaw * (PI / 180.0))
                        + correction * f32::sin(current_yaw * (PI / 180.0));

                    let _ = cb
                        .stability_2_speed_set(
                            x_speed,
                            y_speed,
                            0.0,
                            0.0,
                            initial_yaw,
                            config.depth,
                        )
                        .await;
                    // DelayAction::new(1.0).execute().await;
                }
            }

            Side::Right => {
                if red.len() > 0 {
                    // Center on average x of blue
                    let avg_x = red.iter().map(|d| *d.position().x() as f32).sum::<f32>()
                        / red.len() as f32;

                    #[cfg(feature = "logging")]
                    logln!("AVG X: {}", avg_x);

                    #[cfg(feature = "logging")]
                    logln!("True Count: {}", true_count);

                    if avg_x.abs() > 0.1 {
                        let correction = -0.5 * avg_x;
                        let fwd = 0.0;
                        let x_speed = -fwd * f32::sin(current_yaw * (PI / 180.0))
                            + correction * f32::cos(current_yaw * (PI / 180.0));
                        let y_speed = fwd * f32::cos(current_yaw * (PI / 180.0))
                            + correction * f32::sin(current_yaw * (PI / 180.0));
                        let _ = cb
                            .stability_2_speed_set(
                                x_speed,
                                y_speed,
                                0.0,
                                0.0,
                                initial_yaw,
                                config.depth,
                            )
                            .await;
                    } else {
                        let fwd = config.speed;
                        let correction = 0.0;
                        let x_speed = -fwd * f32::sin(current_yaw * (PI / 180.0))
                            + correction * f32::cos(current_yaw * (PI / 180.0));
                        let y_speed = fwd * f32::cos(current_yaw * (PI / 180.0))
                            + correction * f32::sin(current_yaw * (PI / 180.0));

                        true_count += 1;

                        if true_count > config.true_count {
                            let _ = cb
                                .stability_2_speed_set(
                                    x_speed,
                                    y_speed,
                                    0.0,
                                    0.0,
                                    initial_yaw,
                                    config.depth,
                                )
                                .await;
                            traversal_timer.execute().await;
                            break;
                        }
                    }
                } else {
                    // Fallback search behavior
                    #[cfg(feature = "logging")]
                    logln!("RIGHT: Missing Features, Fallback");
                    // DelayAction::new(1.0).execute().await;
                    let correction = 0.2;
                    let fwd = 0.0;
                    let x_speed = -fwd * f32::sin(current_yaw * (PI / 180.0))
                        + correction * f32::cos(current_yaw * (PI / 180.0));
                    let y_speed = fwd * f32::cos(current_yaw * (PI / 180.0))
                        + correction * f32::sin(current_yaw * (PI / 180.0));

                    let _ = cb
                        .stability_2_speed_set(
                            x_speed,
                            y_speed,
                            0.0,
                            0.0,
                            initial_yaw,
                            config.depth,
                        )
                        .await;
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
    const TIMEOUT: f32 = 30.0;

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
    const TIMEOUT: f32 = 30.0;

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
