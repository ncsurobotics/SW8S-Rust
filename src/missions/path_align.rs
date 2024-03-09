use crate::{
    missions::{
        action::{ActionChain, ActionConcurrent, TupleSecond},
        movement::{AdjustMovement, CountTrue},
        vision::VisionNormOffset,
    },
    vision::{
        buoy::{self, Buoy},
        nn_cv2::OnnxModel,
        path::Path,
        VisualDetector,
    },
};

use super::{
    action::{Action, ActionExec, ActionSequence, ActionWhile},
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::DelayAction,
    movement::{StraightMovement, ZeroMovement},
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

pub fn path_align<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec + '_ {
    const DEPTH: f32 = 1.0;

    let forward_power = 0.3;
    let delay_s = 6.0;

    ActionSequence::new(
        ZeroMovement::new(context, DEPTH),
        ActionChain::new(
            VisionNormOffset::<Con, Path, f64>::new(context, Path::default()),
            TupleSecond::new(ActionConcurrent::new(
                AdjustMovement::new(context, DEPTH),
                CountTrue::new(3),
            )),
        ),
    )
}
