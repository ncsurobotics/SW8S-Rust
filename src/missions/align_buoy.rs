use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{ActionChain, ActionConcurrent, ActionSequence, ActionWhile, TupleSecond},
        basic::descend_and_go_forward,
        extra::{AlwaysTrue, CountFalse, IsSome, OutputType, ToVec},
        movement::{
            ClampX, MultiplyX, OffsetToPose, ReplaceX, Stability2Adjust, Stability2Movement,
            Stability2Pos, StripY, ZeroMovement,
        },
        vision::{ExtractPosition, MidPoint, Norm, Vision},
    },
    vision::{buoy_model::BuoyModel, nn_cv2::OnnxModel, Offset2D},
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
    const DEPTH: f32 = -1.0;
    const FALSE_COUNT: u32 = 7;

    act_nest!(
        ActionSequence::new,
        descend_and_go_forward(context),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            Vision::<Con, BuoyModel<OnnxModel>, f64>::new(context, BuoyModel::default()),
            TupleSecond::<_, bool>::new(ActionConcurrent::new(
                ActionSequence::new(
                    act_nest!(
                        ActionChain::new,
                        ToVec::new(),
                        Norm::new(BuoyModel::default()),
                        ExtractPosition::new(),
                        MidPoint::new(),
                        OffsetToPose::<Offset2D<f64>>::default(),
                        ReplaceX::new(),
                        MultiplyX::new(0.5),
                        ClampX::<Stability2Adjust>::new(0.1),
                        StripY::<Stability2Adjust>::default(),
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
