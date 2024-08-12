use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        extra::ToVec,
        meb::WaitArm,
        movement::{AdjustType, ConstYaw},
        vision::{SizeUnder, VisionNoStrip},
    },
    vision::{
        gate_poles::{GatePoles, Target},
        nn_cv2::{OnnxModel, YoloClass},
        DrawRect2d, Offset2D, VisualDetection,
    },
};

use super::{
    action::{
        Action, ActionChain, ActionConcurrent, ActionExec, ActionMod, ActionSequence, ActionWhile,
    },
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    comms::StartBno055,
    extra::{CountTrue, OutputType},
    movement::{Stability2Adjust, Stability2Movement, Stability2Pos},
    vision::VisionNorm,
};

pub fn coinflip<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec<()> + '_ {
    const TRUE_COUNT: u32 = 2;
    const DELAY_TIME: f32 = 3.0;

    const DEPTH: f32 = -1.25;
    const ALIGN_X_SPEED: f32 = 0.0;
    const ALIGN_Y_SPEED: f32 = 0.0;
    const ALIGN_YAW_SPEED: f32 = 45.0;

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
                VisionNoStrip::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::load_640(0.7),
                ),
                CountTrue::new(TRUE_COUNT),
            ),
        )),
    )
}

#[derive(Debug)]
struct SizePass<T, U> {
    values: Vec<VisualDetection<T, U>>,
}

impl<T, U> Default for SizePass<T, U> {
    fn default() -> Self {
        Self { values: vec![] }
    }
}

impl<T, U> Action for SizePass<T, U> {}

impl<T, U> ActionMod<Vec<VisualDetection<T, U>>> for SizePass<T, U>
where
    T: Send + Sync + Clone,
    U: Send + Sync + Clone,
{
    fn modify(&mut self, input: &Vec<VisualDetection<T, U>>) {
        self.values = input.clone();
    }
}

impl ActionExec<bool> for SizePass<YoloClass<Target>, DrawRect2d> {
    async fn execute(&mut self) -> bool {
        self.values
            .iter()
            .map(|detect| {
                let class = detect.class().identifier();
                let detect_size = detect.position().size();

                let size_min = match class {
                    Target::Red | Target::Blue | Target::Middle => 400.0,
                    Target::Pole => 4_500.0,
                    _ => f64::MAX,
                };

                (detect_size.width * detect_size.height) > size_min
            })
            .any(|x| x)
    }
}
