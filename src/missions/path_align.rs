use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            TupleSecond,
        },
        basic::DelayAction,
        extra::{CountFalse, CountTrue, OutputType, Terminal, ToVec},
        movement::{
            AdjustType, ClampX, FlipX, FlipY, OffsetToPose, SetY, Stability2Adjust,
            Stability2Movement, Stability2Pos, ZeroMovement,
        },
        vision::{
            DetectTarget, ExtractPosition, MidPoint, VisionNormBottom, VisionNormBottomAngle,
        },
    },
    vision::{path::Path, path_cv::PathCV},
};

use super::{
    action::ActionExec,
    action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard},
};

// pub fn path_align<
//     Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
// >(
//     context: &Con,
// ) -> impl ActionExec<()> + '_ {
//     const DEPTH: f32 = -1.25;
//     const PATH_ALIGN_SPEED: f32 = 0.3;

//     act_nest!(
//         ActionSequence::new,
//         ZeroMovement::new(context, DEPTH),
//         DelayAction::new(2.0),
//         ActionWhile::new(ActionChain::new(
//             VisionNormBottomAngle::<Con, PathCV, f64>::new(context, PathCV::default()),
//             TupleSecond::new(ActionConcurrent::new(
//                 act_nest!(
//                     ActionChain::new,
//                     ActionDataConditional::new(
//                         DetectTarget::new(true),
//                         act_nest!(
//                             ActionChain::new,
//                             ExtractPosition::new(),
//                             MidPoint::new(),
//                             OffsetToPose::default(),
//                             ClampX::<Stability2Adjust>::new(0.3),
//                             FlipY::default(),
//                             Stability2Movement::new(
//                                 context,
//                                 Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
//                             ),
//                         ),
//                         act_nest!(
//                             ActionSequence::new,
//                             Terminal::new(),
//                             Stability2Movement::new(
//                                 context,
//                                 Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
//                             ),
//                         ),
//                     ),
//                     OutputType::<()>::new(),
//                 ),
//                 ActionChain::new(DetectTarget::new(true), CountTrue::new(10)),
//             )),
//         )),
//         ActionWhile::new(ActionChain::new(
//             VisionNormBottom::<Con, PathCV, f64>::new(context, PathCV::default()),
//             TupleSecond::new(ActionConcurrent::new(
//                 act_nest!(
//                     ActionChain::new,
//                     ActionDataConditional::new(
//                         DetectTarget::new(true),
//                         act_nest!(
//                             ActionChain::new,
//                             ExtractPosition::new(),
//                             MidPoint::new(),
//                             OffsetToPose::default(),
//                             ClampX::<Stability2Adjust>::new(0.3),
//                             FlipY::default(),
//                             Stability2Movement::new(
//                                 context,
//                                 Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
//                             ),
//                         ),
//                         act_nest!(
//                             ActionSequence::new,
//                             Terminal::new(),
//                             Stability2Movement::new(
//                                 context,
//                                 Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
//                             ),
//                         ),
//                     ),
//                     OutputType::<()>::new(),
//                 ),
//                 ActionChain::new(DetectTarget::new(true), CountFalse::new(10)),
//             )),
//         )),
//         // ActionWhile::new(ActionChain::new(
//         //     VisionNormBottom::<Con, PathCV, f64>::new(context, PathCV::default()),
//         //     TupleSecond::new(ActionConcurrent::new(
//         //         act_nest!(
//         //             ActionChain::new,
//         //             act_nest!(
//         //                 ActionChain::new,
//         //                 ActionDataConditional::new(
//         //                     DetectTarget::new(true),
//         //                     act_nest!(
//         //                         ActionChain::new,
//         //                         ExtractPosition::new(),
//         //                         MidPoint::new(),
//         //                         OffsetToPose::default(),
//         //                         LinearYawFromX::<Stability2Adjust>::default(),
//         //                         Stability2Movement::new(
//         //                             context,
//         //                             Stability2Pos::new(
//         //                                 0.0,
//         //                                 PATH_ALIGN_SPEED / 2.0,
//         //                                 0.0,
//         //                                 0.0,
//         //                                 None,
//         //                                 DEPTH
//         //                             ),
//         //                         ),
//         //                     ),
//         //                     act_nest!(
//         //                         ActionSequence::new,
//         //                         Terminal::new(),
//         //                         Stability2Movement::new(
//         //                             context,
//         //                             Stability2Pos::new(
//         //                                 0.0,
//         //                                 PATH_ALIGN_SPEED / 2.0,
//         //                                 0.0,
//         //                                 0.0,
//         //                                 None,
//         //                                 DEPTH
//         //                             ),
//         //                         ),
//         //                     ),
//         //                 OutputType::<()>::new(),
//         //             ),
//         //             act_nest!(
//         //                 ActionSequence::new,
//         //                 Terminal::new(),
//         //                 Stability2Movement::new(
//         //                     context,
//         //                     Stability2Pos::new(0.0, PATH_ALIGN_SPEED / 1.5, 0.0, 0.0, None, DEPTH),
//         //                 ),
//         //             ),
//         //             OutputType::<()>::new(),
//         //         ),
//         //         ActionChain::new(DetectTarget::new(true), CountFalse::new(10)),
//         //     )),
//         // )),
//         Terminal::new(),
//     )
// }

pub async fn path_align_procedural<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
) {
    const DEPTH: f32 = -1.25;
    const PATH_ALIGN_SPEED: f32 = 0.3;
    const DETECTIONS: u8 = 10;

    #[cfg(feature = "logging")]
    logln!("Starting path align");

    let cb = context.get_control_board();
    let mut vision_norm_bottom =
        VisionNormBottomAngle::<Con, PathCV, f64>::new(context, PathCV::default());

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw() as f32;
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    cb.stability_2_speed_set(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, initial_yaw, DEPTH)
        .await;

    let mut last_set_yaw = initial_yaw;
    let mut consec_detections = 0;

    #[cfg(feature = "logging")]
    logln!("Starting path detection");

    loop {
        if consec_detections >= DETECTIONS {
            #[cfg(feature = "logging")]
            logln!("Finished path align");

            break;
        }

        if let Some(current_angle) = cb.responses().get_angles().await {
            let current_yaw = *current_angle.yaw() as f32;

            // For the opencv impl of path detection, the returned vector is guaranteed to contain 1 item
            let detections = vision_norm_bottom.execute().await.unwrap_or_else(|e| {
                #[cfg(feature = "logging")]
                logln!(
                    "Getting path detection resulted in error: `{e}`\n\tUsing empty detection vec"
                );
                vec![]
            });

            let mut positions = detections
                .into_iter()
                .filter_map(|d| d.class().then_some(d.position().clone()));

            if let Some(position) = positions.next() {
                let x = *position.x() as f32;
                let y = (*position.y() as f32) * -1.0;
                let angle = position.angle();
                last_set_yaw = current_yaw + *angle as f32;
                cb.stability_2_speed_set(x, y, 0.0, 0.0, last_set_yaw, DEPTH)
                    .await;
                consec_detections += 1;
            } else {
                cb.stability_2_speed_set(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, last_set_yaw, DEPTH)
                    .await;
                consec_detections = 0;
            }
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get current angle");
        }

        #[cfg(feature = "logging")]
        logln!("Positive detection count: {consec_detections}");
    }
}
