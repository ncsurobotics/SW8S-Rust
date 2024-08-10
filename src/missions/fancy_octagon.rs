use opencv::core::Size;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            RaceAction, TupleSecond,
        },
        basic::DelayAction,
        extra::{
            AlwaysBetterFalse, AlwaysBetterTrue, AlwaysTrue, CountFalse, CountTrue, OutputType,
            Terminal, ToVec,
        },
        movement::{
            AdjustType, ClampX, ConstYaw, LinearYawFromX, NoAdjust, OffsetToPose, SetX,
            Stability2Adjust, Stability2Movement, Stability2Pos, StripY, ZeroMovement,
        },
        vision::{DetectTarget, ExtractPosition, MidPoint, Norm, Vision},
    },
    vision::{
        path::{Path, Yuv},
        Offset2D,
    },
    POOL_YAW_SIGN,
};

use super::{
    action::ActionExec,
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
};

pub fn octagon_path_model() -> Path {
    Path::new(
        (Yuv {
            y: 64,
            u: 127,
            v: 127,
        })..=(Yuv {
            y: 180,
            u: 224,
            v: 224,
        }),
        5.0..=200.0,
        4,
        Size::from((400, 300)),
        3,
    )
}

pub fn fancy_octagon<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    const FULL_SPEED_Y: f32 = 0.7;
    const FULL_SPEED_X: f32 = 0.1;
    const FULL_SPEED_PITCH: f32 = -45.0 / 4.0;
    const DEPTH: f32 = -0.75;

    const INIT_X: f32 = 0.0;
    const INIT_Y: f32 = 0.0;
    const INIT_TIME: f32 = 3.0;

    const BLIND_TIME: f32 = 3.0;

    const X_CLAMP: f32 = 0.3;

    const FALSE_COUNT: u32 = 3;
    const ADJUST_COUNT: u32 = 2;

    const OCTAGON_SPIN: f32 = 80.0 * POOL_YAW_SIGN;

    const MISSION_END_TIME: f32 = INIT_TIME + BLIND_TIME + 13.0;

    const ALIGN_YAW_SPEED: f32 = 5.0 * POOL_YAW_SIGN;

    act_nest!(
        ActionSequence::new,
        ActionWhile::new(act_nest!(
            ActionSequence::new,
            act_nest!(
                ActionChain::new,
                NoAdjust::<Stability2Adjust>::new(),
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(OCTAGON_SPIN)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            DelayAction::new(1.0),
            ActionChain::<bool, _, _>::new(AlwaysTrue::default(), CountTrue::new(ADJUST_COUNT)),
        ),),
        DelayAction::new(2.0),
        ActionChain::new(
            Stability2Movement::new(
                context,
                Stability2Pos::new(INIT_X, INIT_Y, 0.0, 0.0, None, DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(INIT_TIME),
        act_nest!(
            ActionChain::new,
            Stability2Movement::new(
                context,
                Stability2Pos::new(
                    FULL_SPEED_X,
                    FULL_SPEED_Y,
                    FULL_SPEED_PITCH,
                    0.0,
                    None,
                    DEPTH
                )
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(MISSION_END_TIME),
        ActionWhile::new(ActionSequence::new(
            act_nest!(
                ActionChain::new,
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 0.0, FULL_SPEED_PITCH, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            act_nest!(
                ActionChain::new,
                Vision::<Con, Path, f64>::new(context, octagon_path_model()),
                DetectTarget::new(true),
                CountTrue::new(3),
            ),
        ),),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            Vision::<Con, Path, f64>::new(context, octagon_path_model()),
            ActionDataConditional::new(
                DetectTarget::new(true),
                ActionSequence::new(
                    act_nest!(
                        ActionChain::new,
                        Norm::new(Path::default()),
                        ExtractPosition::new(),
                        MidPoint::new(),
                        OffsetToPose::<Offset2D<f64>>::default(),
                        LinearYawFromX::<Stability2Adjust>::new(7.0),
                        ClampX::<Stability2Adjust>::new(X_CLAMP),
                        StripY::<Stability2Adjust>::new(),
                        ActionChain::new(
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(
                                    FULL_SPEED_X,
                                    FULL_SPEED_Y,
                                    0.0,
                                    0.0,
                                    None,
                                    DEPTH
                                )
                            ),
                            OutputType::<()>::new(),
                        ),
                    ),
                    AlwaysBetterTrue::new(),
                ),
                ActionSequence::new(
                    act_nest!(
                        ActionSequence::new,
                        Terminal::new(),
                        SetX::<Stability2Adjust>::new(AdjustType::Replace(FULL_SPEED_X)),
                        StripY::<Stability2Adjust>::new(),
                        ActionChain::new(
                            Stability2Movement::new(
                                context,
                                Stability2Pos::new(
                                    FULL_SPEED_X,
                                    FULL_SPEED_Y,
                                    0.0,
                                    0.0,
                                    None,
                                    DEPTH
                                )
                            ),
                            OutputType::<()>::new(),
                        ),
                    ),
                    AlwaysBetterFalse::new(),
                ),
            ),
            CountFalse::new(FALSE_COUNT)
        ),),
        ZeroMovement::new(context, DEPTH),
        OutputType::<()>::new()
    )
}
