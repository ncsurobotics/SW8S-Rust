use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    comms::meb::{MainElectronicsBoard, MebCmd},
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            TupleSecond,
        },
        basic::DelayAction,
        extra::{AlwaysTrue, CountFalse, CountTrue, IsSome, OutputType, Terminal},
        movement::{
            AdjustType, ClampX, ConstYaw, Descend, LinearYawFromX, MultiplyX, OffsetToPose,
            ReplaceX, SetX, SetY, SideMult, Stability2Adjust, Stability2Movement, Stability2Pos,
            StraightMovement, StripY, ZeroMovement,
        },
        vision::{DetectTarget, ExtractPosition, MidPoint, Norm, Vision},
    },
    vision::{
        buoy_model::{BuoyModel, Target},
        nn_cv2::OnnxModel,
        Offset2D,
    },
};

use super::{
    action::{Action, ActionExec},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
};

#[derive(Debug)]
pub struct FireTorpedo<'a, T> {
    meb: &'a T,
}

impl<'a, T> FireTorpedo<'a, T> {
    pub fn new(meb: &'a T) -> Self {
        Self { meb }
    }
}

impl<T> Action for FireTorpedo<'_, T> {}

impl<T: GetMainElectronicsBoard> ActionExec<()> for FireTorpedo<'_, T> {
    async fn execute<'a>(&'a mut self) {
        let send_cmd = |meb: &'a MainElectronicsBoard<WriteHalf<SerialStream>>, cmd| async move {
            match meb.send_msg(cmd).await {
                Ok(()) => println!("{:#?} success", cmd),
                Err(e) => eprintln!("{:#?} failure: {:#?}", cmd, e),
            };
        };

        let meb = self.meb.get_main_electronics_board();
        for _ in 0..3 {
            send_cmd(meb, MebCmd::T1Trig).await;
            send_cmd(meb, MebCmd::T2Trig).await;
        }
    }
}
