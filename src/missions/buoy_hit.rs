use crate::vision::{
    buoy::{Buoy, Target},
    nn_cv2::{VisionModel, YoloDetection},
    yolo_model::YoloProcessor,
    VisualDetector,
};

use super::{
    action::{Action, ActionExec, ActionMod},
    action_context::{GetControlBoard, GetFrontCamMat},
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use futures::stream::ForEach;
use std::sync::mpsc::Iter;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio_serial::SerialStream;

#[derive(Debug)]
struct DriveToBuoyVision<'a, T, U: VisionModel> {
    context: &'a T,
    buoy_model: Buoy<U>,
    target_depth: f32,
    forward_power: f32,
}

impl<T, U: VisionModel> Action for DriveToBuoyVision<'_, T, U> where U: VisionModel {}

impl<T, U> ActionMod<f32> for DriveToBuoyVision<'_, T, U>
where
    U: VisionModel,
{
    // TODO: Do we even need this?
    fn modify(&mut self, input: f32) {
        self.target_depth = input;
    }
}

#[async_trait]
impl<T, U> ActionExec for DriveToBuoyVision<'_, T, U>
where
    T: GetControlBoard<WriteHalf<SerialStream>> + GetFrontCamMat + Sync + Unpin,
    U: VisionModel,
{
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        let mut buoy_model = Buoy::default();
        let class_of_interest = Target::Abydos1;
        println!("Getting control board and setting speed to zero before buoy search.");
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await?;
        println!("GOT ZERO SPEED SET");
        while true {
            let camera_aquisition = self.context.get_front_camera_mat();
            let model_acquisition = buoy_model.detect(&camera_aquisition.await);
            match model_acquisition {
                Ok(acquisition_vec) => {
                    let detected_item = acquisition_vec
                        .iter()
                        .find(|&result| *result.class() == class_of_interest);
                    match (detected_item) {
                        Some(scan) => {
                            let position = buoy_model.normalize(scan.position());
                            self.context
                                .get_control_board()
                                .stability_2_speed_set_initial_yaw(
                                    self.forward_power,
                                    position.x as f32,
                                    0.0,
                                    0.0,
                                    self.target_depth,
                                )
                                .await?;
                        }
                        None => todo!(),
                    }
                }
                Err(_) => return Ok(()),
            }
        }
        Ok(())
    }
}
