use std::{iter::Sum, ops::Div};

use anyhow::{anyhow, bail};
use async_trait::async_trait;

use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    vision::{
        gate_poles::{GatePoles, Target},
        nn_cv2::{OnnxModel, YoloClass},
        Offset2D, VisualDetection,
    },
};

use super::{
    action::{
        wrap_action, Action, ActionChain, ActionConcurrent, ActionExec, ActionMod, ActionSequence,
        ActionWhile, FirstValid, TupleSecond,
    },
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
    basic::descend_and_go_forward,
    comms::StartBno055,
    movement::{CountFalse, CountTrue},
    vision::VisionNorm,
};

pub fn gate_run_complex<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
>(
    context: &Con,
) -> impl ActionExec + '_ {
    let depth: f32 = -1.0;

    ActionSequence::new(
        ActionConcurrent::new(descend_and_go_forward(context), StartBno055::new(context)),
        ActionSequence::new(
            adjust_logic(context, depth, CountTrue::new(3)),
            adjust_logic(context, depth, CountFalse::new(10)),
        ),
    )
}

pub fn adjust_logic<
    'a,
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat,
    X: 'a
        + ActionMod<anyhow::Result<Vec<VisualDetection<YoloClass<Target>, Offset2D<f64>>>>>
        + ActionExec<Output = anyhow::Result<()>>,
>(
    context: &'a Con,
    _depth: f32,
    end_condition: X,
) -> impl ActionExec + 'a {
    ActionWhile::new(ActionChain::new(
        VisionNorm::<Con, GatePoles<OnnxModel>, f64>::new(context, GatePoles::default()),
        TupleSecond::new(ActionConcurrent::new(
            act_nest!(
                wrap_action(ActionConcurrent::new, FirstValid::new),
                DetectTarget::new(YoloClass {
                    identifier: Target::Earth,
                    confidence: 1.0,
                }),
                DetectTarget::new(YoloClass {
                    identifier: Target::Abydos,
                    confidence: 1.0,
                }),
                DetectTarget::new(YoloClass {
                    identifier: Target::LargeGate,
                    confidence: 1.0,
                }),
            ),
            end_condition,
        )),
    ))
}

#[derive(Debug)]
pub struct DetectTarget<T, U> {
    results: anyhow::Result<Vec<VisualDetection<T, U>>>,
    target: T,
}

impl<T, U> DetectTarget<T, U> {
    pub const fn new(target: T) -> Self {
        Self {
            results: Ok(vec![]),
            target,
        }
    }
}

impl<T, U> Action for DetectTarget<T, U> {}

#[async_trait]
impl<T: Send + Sync + PartialEq, U: Send + Sync + Clone + Sum<U> + Div<usize, Output = U>>
    ActionExec for DetectTarget<T, U>
{
    type Output = anyhow::Result<U>;
    async fn execute(&mut self) -> Self::Output {
        if let Ok(vals) = &self.results {
            let passing_vals: Vec<_> = vals
                .iter()
                .filter(|entry| entry.class() == &self.target)
                .collect();
            if !passing_vals.is_empty() {
                Ok(passing_vals
                    .iter()
                    .map(|entry| entry.position().clone())
                    .sum::<U>()
                    / passing_vals.len())
            } else {
                bail!("No valid detections")
            }
        } else {
            bail!("Empty results")
        }
    }
}

impl<T: Send + Sync + Clone, U: Send + Sync + Clone>
    ActionMod<anyhow::Result<Vec<VisualDetection<T, U>>>> for DetectTarget<T, U>
{
    fn modify(&mut self, input: &anyhow::Result<Vec<VisualDetection<T, U>>>) {
        self.results = input
            .as_ref()
            .map(|valid| valid.clone())
            .map_err(|invalid| anyhow!("{}", invalid));
    }
}
