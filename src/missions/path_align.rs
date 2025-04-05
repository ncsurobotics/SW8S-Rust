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
            AdjustType, ClampX, FlipX, FlipY, LinearYawFromX, OffsetToPose, SetY, Stability2Adjust,
            Stability2Movement, Stability2Pos, ZeroMovement,
        },
        vision::{DetectTarget, ExtractPosition, MidPoint, VisionNormBottom},
    },
    vision::path::Path,
    vision::path_cv::PathCV,
};

use super::{
    action::ActionExec,
    action_context::{GetBottomCamMat, GetControlBoard, GetMainElectronicsBoard},
};

pub fn path_align<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetBottomCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const DEPTH: f32 = -1.25;
    const PATH_ALIGN_SPEED: f32 = 0.3;

    act_nest!(
        ActionSequence::new,
        ZeroMovement::new(context, DEPTH),
        DelayAction::new(2.0),
        ActionWhile::new(ActionChain::new(
            VisionNormBottom::<Con, PathCV, f64>::new(context, PathCV::default()),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ActionDataConditional::new(
                        DetectTarget::new(true),
                        act_nest!(
                            ActionChain::new,
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::default(),
                            LinearYawFromX::<Stability2Adjust>::new(1.0),
                            ClampX::<Stability2Adjust>::new(0.3),
                            FlipY::default(),
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
                            ),
                        ),
                        act_nest!(
                            ActionSequence::new,
                            Terminal::new(),
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
                            ),
                        ),
                    ),
                    OutputType::<()>::new(),
                ),
                ActionChain::new(DetectTarget::new(true), CountTrue::new(10)),
            )),
        )),
        ActionWhile::new(ActionChain::new(
            VisionNormBottom::<Con, PathCV, f64>::new(context, PathCV::default()),
            TupleSecond::new(ActionConcurrent::new(
                act_nest!(
                    ActionChain::new,
                    ActionDataConditional::new(
                        DetectTarget::new(true),
                        act_nest!(
                            ActionChain::new,
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::default(),
                            LinearYawFromX::<Stability2Adjust>::new(1.0),
                            ClampX::<Stability2Adjust>::new(0.3),
                            FlipY::default(),
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
                            ),
                            // SetY::<Stability2Adjust>::new(AdjustType::Replace(PATH_ALIGN_SPEED)),
                        ),
                        act_nest!(
                            ActionSequence::new,
                            Terminal::new(),
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(0.0, PATH_ALIGN_SPEED, 0.0, 0.0, None, DEPTH),
                            ),
                        ),
                    ),
                    OutputType::<()>::new(),
                ),
                ActionChain::new(DetectTarget::new(true), CountFalse::new(10)),
            )),
        )),
        // ActionWhile::new(ActionChain::new(
        //     VisionNormBottom::<Con, PathCV, f64>::new(context, PathCV::default()),
        //     TupleSecond::new(ActionConcurrent::new(
        //         act_nest!(
        //             ActionChain::new,
        //             act_nest!(
        //                 ActionChain::new,
        //                 ActionDataConditional::new(
        //                     DetectTarget::new(true),
        //                     act_nest!(
        //                         ActionChain::new,
        //                         ExtractPosition::new(),
        //                         MidPoint::new(),
        //                         OffsetToPose::default(),
        //                         LinearYawFromX::<Stability2Adjust>::default(),
        //                         Stability2Movement::new(
        //                             context,
        //                             Stability2Pos::new(
        //                                 0.0,
        //                                 PATH_ALIGN_SPEED / 2.0,
        //                                 0.0,
        //                                 0.0,
        //                                 None,
        //                                 DEPTH
        //                             ),
        //                         ),
        //                     ),
        //                     act_nest!(
        //                         ActionSequence::new,
        //                         Terminal::new(),
        //                         Stability2Movement::new(
        //                             context,
        //                             Stability2Pos::new(
        //                                 0.0,
        //                                 PATH_ALIGN_SPEED / 2.0,
        //                                 0.0,
        //                                 0.0,
        //                                 None,
        //                                 DEPTH
        //                             ),
        //                         ),
        //                     ),
        //                 OutputType::<()>::new(),
        //             ),
        //             act_nest!(
        //                 ActionSequence::new,
        //                 Terminal::new(),
        //                 Stability2Movement::new(
        //                     context,
        //                     Stability2Pos::new(0.0, PATH_ALIGN_SPEED / 1.5, 0.0, 0.0, None, DEPTH),
        //                 ),
        //             ),
        //             OutputType::<()>::new(),
        //         ),
        //         ActionChain::new(DetectTarget::new(true), CountFalse::new(10)),
        //     )),
        // )),
        Terminal::new(),
    )
}
