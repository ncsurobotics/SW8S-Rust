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
        comms::StartBno055,
        extra::{AlwaysTrue, CountFalse, CountTrue, IsSome, OutputType, Terminal},
        fire_torpedo::FireTorpedo,
        movement::{
            AdjustType, ClampX, ConstYaw, LinearYawFromX, MultiplyX, OffsetToPose, ReplaceX, SetX,
            SetY, Stability2Adjust, Stability2Movement, Stability2Pos, ZeroMovement,
        },
        vision::{
            DetectTarget, ExtractPosition, MidPoint, Norm, SizeUnder, Vision, VisionSizeLock,
        },
    },
    vision::{
        buoy_model::{BuoyModel, Target},
        nn_cv2::OnnxModel,
        Offset2D,
    },
};

use super::{
    action::ActionExec,
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
};

pub fn buoy_align<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    const Y_SPEED: f32 = 0.2;
    const Y_SPEED_FAST: f32 = 0.5;
    const DEPTH: f32 = -1.0;
    const FALSE_COUNT: u32 = 5;

    const ALIGN_X_SPEED: f32 = 0.0;
    const ALIGN_Y_SPEED: f32 = 0.0;
    const ALIGN_YAW_SPEED: f32 = 4.0;

    const FAST_DISTANCE: f64 = 3_000.0;
    const CORRECT_YAW_SPEED: f32 = 3.0;
    const CORRECT_X_MULTIPLY: f32 = 0.5;
    const CORRECT_X_CLAMP: f32 = 0.15;

    act_nest!(
        ActionSequence::new,
        StartBno055::new(context),
        act_nest!(
            ActionChain::new,
            Stability2Movement::new(context, Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)),
            OutputType::<()>::new(),
        ),
        DelayAction::new(2.0),
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
                Vision::<Con, BuoyModel<OnnxModel>, f64>::new(context, BuoyModel::default()),
                IsSome::default(),
                CountTrue::new(2)
            )
        )),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            Vision::<Con, BuoyModel<OnnxModel>, f64>::new(context, BuoyModel::default()),
            TupleSecond::<_, bool>::new(ActionConcurrent::new(
                ActionSequence::new(
                    act_nest!(
                        ActionChain::new,
                        ActionDataConditional::new(
                            DetectTarget::new(Target::Buoy),
                            ActionDataConditional::new(
                                SizeUnder::new(FAST_DISTANCE),
                                act_nest!(
                                    ActionChain::new,
                                    Norm::new(BuoyModel::default()),
                                    ExtractPosition::new(),
                                    MidPoint::new(),
                                    OffsetToPose::<Offset2D<f64>>::default(),
                                    ReplaceX::new(),
                                    LinearYawFromX::<Stability2Adjust>::new(CORRECT_YAW_SPEED),
                                    MultiplyX::new(CORRECT_X_MULTIPLY),
                                    ClampX::<Stability2Adjust>::new(CORRECT_X_CLAMP),
                                    SetY::<Stability2Adjust>::new(AdjustType::Replace(
                                        Y_SPEED_FAST
                                    )),
                                ),
                                act_nest!(
                                    ActionChain::new,
                                    Norm::new(BuoyModel::default()),
                                    ExtractPosition::new(),
                                    MidPoint::new(),
                                    OffsetToPose::<Offset2D<f64>>::default(),
                                    ReplaceX::new(),
                                    LinearYawFromX::<Stability2Adjust>::new(CORRECT_YAW_SPEED),
                                    MultiplyX::new(CORRECT_X_MULTIPLY),
                                    ClampX::<Stability2Adjust>::new(CORRECT_X_CLAMP),
                                    SetY::<Stability2Adjust>::new(AdjustType::Replace(Y_SPEED)),
                                )
                            ),
                            act_nest!(
                                ActionSequence::new,
                                Terminal::new(),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(0.0)),
                                SetX::<Stability2Adjust>::new(AdjustType::Replace(0.1)),
                            )
                        ),
                        Stability2Movement::new(
                            context,
                            Stability2Pos::new(0.0, Y_SPEED, 0.0, 0.0, None, DEPTH)
                        ),
                        OutputType::<()>::new(),
                    ),
                    AlwaysTrue::new()
                ),
                ActionChain::new(IsSome::default(), CountFalse::new(FALSE_COUNT))
            )),
        ),),
        ZeroMovement::new(context, DEPTH),
        OutputType::<()>::new()
    )
}

pub fn buoy_align_shot<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    const Y_SPEED: f32 = 0.2;
    const DEPTH: f32 = -0.9;
    const TRUE_COUNT: u32 = 2;
    const FALSE_COUNT: u32 = 5;

    const BACKUP_Y_SPEED: f32 = -0.5;
    const BACKUP_TIME: f32 = 6.0;

    const ALIGN_X_SPEED: f32 = 0.0;
    const ALIGN_Y_SPEED: f32 = 0.0;
    const ALIGN_YAW_SPEED: f32 = 3.0;

    const SHOT_DEPTH: f32 = -0.6;
    const SHOT_ANGLE: f32 = 22.5;

    act_nest!(
        ActionSequence::new,
        act_nest!(
            ActionChain::new,
            Stability2Movement::new(context, Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)),
            OutputType::<()>::new(),
        ),
        DelayAction::new(4.0),
        act_nest!(
            ActionChain::new,
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, BACKUP_Y_SPEED, 0.0, 0.0, None, DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(BACKUP_TIME),
        act_nest!(
            ActionChain::new,
            Stability2Movement::new(context, Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)),
            OutputType::<()>::new(),
        ),
        DelayAction::new(4.0),
        ActionWhile::new(ActionSequence::new(
            act_nest!(
                ActionChain::new,
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(-ALIGN_X_SPEED, ALIGN_Y_SPEED, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            act_nest!(
                ActionChain::new,
                Vision::<Con, BuoyModel<OnnxModel>, f64>::new(context, BuoyModel::default()),
                IsSome::default(),
                CountTrue::new(TRUE_COUNT)
            )
        )),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            VisionSizeLock::<Con, BuoyModel<OnnxModel>, f64>::new(context, BuoyModel::default()),
            TupleSecond::<_, bool>::new(ActionConcurrent::new(
                ActionSequence::new(
                    act_nest!(
                        ActionChain::new,
                        ActionDataConditional::new(
                            DetectTarget::new(Target::Buoy),
                            act_nest!(
                                ActionChain::new,
                                Norm::new(BuoyModel::default()),
                                ExtractPosition::new(),
                                MidPoint::new(),
                                OffsetToPose::<Offset2D<f64>>::default(),
                                ReplaceX::new(),
                                LinearYawFromX::<Stability2Adjust>::new(3.0),
                                MultiplyX::new(0.5),
                                ClampX::<Stability2Adjust>::new(0.15),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(Y_SPEED)),
                            ),
                            act_nest!(
                                ActionSequence::new,
                                Terminal::new(),
                                SetY::<Stability2Adjust>::new(AdjustType::Replace(0.0)),
                                SetX::<Stability2Adjust>::new(AdjustType::Replace(0.0)),
                            )
                        ),
                        Stability2Movement::new(
                            context,
                            Stability2Pos::new(0.0, Y_SPEED, 0.0, 0.0, None, DEPTH)
                        ),
                        OutputType::<()>::new(),
                    ),
                    AlwaysTrue::new()
                ),
                ActionChain::new(IsSome::default(), CountFalse::new(FALSE_COUNT))
            )),
        ),),
        act_nest!(
            ActionChain::new,
            ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
            Stability2Movement::new(
                context,
                Stability2Pos::new(-0.2, 0.0, 0.0, 0.0, None, DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(0.5),
        act_nest!(
            ActionChain::new,
            ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, 0.0, SHOT_ANGLE, 0.0, None, SHOT_DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(1.0),
        FireTorpedo::new(context),
        act_nest!(
            ActionChain::new,
            ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(ALIGN_YAW_SPEED)),
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.2, 0.0, SHOT_ANGLE, 0.0, None, SHOT_DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(1.0),
        FireTorpedo::new(context),
        DelayAction::new(3.0),
        OutputType::<()>::new()
    )
}
