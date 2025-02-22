use tokio::io::WriteHalf;
use tokio_serial::SerialStream;
use rand::Rng;

use crate::{
    act_nest,
    missions::{
        meb::WaitArm,
        movement::{AdjustType, ConstYaw},
    },
    vision::{gate_poles::GatePoles, nn_cv2::OnnxModel},
};

use super::{
    action::{ActionChain, ActionConcurrent, ActionExec, ActionSequence, ActionWhile},
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
    const ALIGN_YAW_SPEED: f32 = 3.0;


    //simulate the coin flip (0 for heads, 1 for tails)
    let coin_flip_result = rand::thread_rng().gen_range(0..2); 
    

    // 90 - AUV starts parallel to the gate (Heads)
    // 180 - AUV tail pointed toward the gate (Tails)
    let initial_yaw: Option<f32> = if coin_flip_result == 0 {90.0} else {180.0};
    /*let (initial_yaw, description) = match coin_flip_result {
        initial_yaw = if coin_flip_result == 0 { 90.0 } else { 180.0 },
            0 => (90.0, "AUV starts parallel to the gate (Heads)"),
            1 => (180.0, "AUV tail pointed toward the gate (Tails)"),
            _ => unreachable!(),  // This line will never execute, but included for exhaustiveness
        };*/

    
    // execute action sequence based on the coin flip outcome. 
    act_nest!(
        ActionSequence::new,
        ActionConcurrent::new(WaitArm::new(context), StartBno055::new(context)),
        ActionChain::new(
            Stability2Movement::new(context, Stability2Pos::new(0.0, 0.0, 0.0, 0.0, initial_yaw, DEPTH)),
            OutputType::<()>::new()
        ),
        //Delay to allow the system to stabilize after setting the initial yaw.
        DelayAction::new(DELAY_TIME),
        //Perform orientation adjustment start a mission. 
        ActionWhile::new(ActionSequence::new(


            act_nest!(
                ActionChain::new,
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(initial_yaw)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),

            //initialize computer vision model
            act_nest!(
                ActionChain::new,
                VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(
                    context,
                    GatePoles::load_640(0.7),
                ),
                CountTrue::new(TRUE_COUNT),
            ),
        )),
    )
}
