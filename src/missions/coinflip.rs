use itertools::Itertools;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        extra::AlwaysTrue,
        meb::WaitArm,
        movement::{AdjustType, ConstYaw},
    },
    vision::{
        gate_poles::{GatePoles, Target},
        nn_cv2::{OnnxModel, YoloClass},
        Offset2D,
    },
};

use super::{
    action::{
        wrap_action, ActionChain, ActionConcurrent, ActionExec, ActionSequence, ActionWhile,
        FirstValid,
    },
    action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard},
    basic::DelayAction,
    comms::StartBno055,
    extra::{CountTrue, OutputType},
    movement::{Stability2Adjust, Stability2Movement, Stability2Pos},
    vision::{DetectTarget, VisionNorm},
};

pub async fn coinflip_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
) {
    #[cfg(feature = "logging")]
    logln!("Starting path align");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;
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
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, 0.0)
        .await;

    loop {
        let detections = vision.execute().await.unwrap_or_else(|e| {
            #[cfg(feature = "logging")]
            logln!("Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec");
            vec![]
        });

        let gate = detections
            .iter()
            .filter(|d| matches!(d.class().identifier, Target::Gate))
            .collect_vec();
    }
}

pub fn coinflip<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + FrontCamIO,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const TRUE_COUNT: u32 = 4;
    const DELAY_TIME: f32 = 3.0;

    const DEPTH: f32 = -1.25;
    const ALIGN_X_SPEED: f32 = 0.0;
    const ALIGN_Y_SPEED: f32 = 0.0;
    const ALIGN_YAW_SPEED: f32 = -3.0;
    const ALIGN_YAW_CORRECTION_SPEED: f32 = 0.0;

    act_nest!(
        ActionSequence::new,
        ActionConcurrent::new(WaitArm::new(context), StartBno055::new(context)),
        ActionChain::new(
            Stability2Movement::new(context, Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)),
            OutputType::<()>::new()
        ),
        DelayAction::new(DELAY_TIME),
        ActionWhile::new(ActionSequence::new(
            act_nest!(
                ActionChain::new,
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(ALIGN_X_SPEED, ALIGN_Y_SPEED, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            act_nest!(
                ActionChain::new,
                VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::load_640(0.8),
                ),
                act_nest!(
                    wrap_action(ActionConcurrent::new, FirstValid::new),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Blue),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Middle),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Red),
                    DetectTarget::<Target, YoloClass<Target>, Offset2D<f64>>::new(Target::Pole),
                ),
                CountTrue::new(TRUE_COUNT),
            ),
        )),
        ActionWhile::new(act_nest!(
            ActionSequence::new,
            act_nest!(
                ActionChain::new,
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_CORRECTION_SPEED)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(ALIGN_X_SPEED, ALIGN_Y_SPEED, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            DelayAction::new(2.0),
            ActionChain::<bool, _, _>::new(AlwaysTrue::new(), CountTrue::new(2),),
        ))
    )
}
