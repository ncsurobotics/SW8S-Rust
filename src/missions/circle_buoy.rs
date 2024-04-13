use crate::{
    act_nest,
    missions::{
        action::{ActionChain, ActionWhile},
        extra::{AlwaysTrue, OutputType, ToVec, Transform},
        movement::{
            aggressive_yaw_from_x, FlatX, LinearYawFromX, OffsetToPose, Stability1Adjust,
            Stability1Movement, Stability1Pos, Stability2Adjust, Stability2Movement, Stability2Pos,
            StripX, StripY,
        },
        vision::{Average, DetectTarget, ExtractPosition, VisionNorm},
    },
    vision::{
        path::{Path, Yuv},
        Offset2D,
    },
};

use super::{
    action::{Action, ActionExec, ActionMod, ActionSequence},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    movement::ZeroMovement,
};

use async_trait::async_trait;

use opencv::core::Size;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

pub fn buoy_circle_sequence<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const DEPTH: f32 = -0.5;

    let lateral_power = 0.3;
    let delay_s = 1.0;
    // Create a DelayAction with hardcoded delay
    let delay_action = DelayAction::new(delay_s);

    // Create the inner ActionSequence
    ActionSequence::new(
        ZeroMovement::new(context, DEPTH),
        ActionSequence::new(
            delay_action.clone(),
            ActionWhile::new(ActionSequence::new(
                act_nest!(
                    ActionChain::new,
                    VisionNorm::<Con, Path, f64>::new(
                        context,
                        Path::new(
                            (Yuv { y: 0, u: 0, v: 128 })..=(Yuv {
                                y: 255,
                                u: 127,
                                v: 255,
                            }),
                            20.0..=800.0,
                            10,
                            Size::from((400, 300)),
                            3,
                        )
                    ),
                    DetectTarget::<bool, bool, Offset2D<f64>>::new(true),
                    ToVec::new(),
                    ExtractPosition::new(),
                    Average::new(),
                    OffsetToPose::default(),
                    Transform::new(Stability2Adjust::default(), |input| aggressive_yaw_from_x(
                        input, 40.0
                    )),
                    StripY::default(),
                    FlatX::default(),
                    Stability2Movement::new(
                        context,
                        Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)
                    ),
                    OutputType::<()>::new()
                ),
                AlwaysTrue::new(),
            )),
        ),
    )
}
