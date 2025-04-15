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

    let mut visionNormBottom =
        VisionNormBottomAngle::<Con, PathCV, f64>::new(context, PathCV::default());
    loop {
        let detections = visionNormBottom.execute().await.unwrap();
        if let Some(detection) = detections
            .into_iter()
            .filter_map(|d| {
                if *d.class() {
                    Some(d.position().clone())
                } else {
                    None
                }
            })
            .next()
        {
            let x = *detection.x() as f32;
            let y = (*detection.y() as f32) * -1.0;
            let angle = detection.angle();
            let cb = context.get_control_board();
            if let Some(current_angle) = cb.responses().get_angles().await {
                cb.stability_2_speed_set(
                    x,
                    y,
                    0.0,
                    0.0,
                    (*current_angle.yaw() as f32) + *angle as f32,
                    DEPTH,
                )
                .await;
            }
        }
    }
}
