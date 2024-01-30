use crate::vision::{buoy::Buoy, VisualDetector};

use super::{
    action::{Action, ActionExec},
    action_context::{GetControlBoard, GetFrontCamMat},
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;

use tokio_serial::SerialStream;

/// Action to drive to a Buoy using vision
/// will not set the power to zero on its own.
#[derive(Debug)]
struct DriveToBuoyVision<'a, T> {
    context: &'a T,
    target_depth: f32,
    forward_power: f32,
}

impl<T> Action for DriveToBuoyVision<'_, T> {}

#[async_trait]
impl<T> ActionExec for DriveToBuoyVision<'_, T>
where
    T: GetControlBoard<WriteHalf<SerialStream>> + GetFrontCamMat + Sync + Unpin,
{
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        let mut buoy_model = Buoy::default();
        let class_of_interest = self.context.get_desired_buoy_gate().await;
        println!("Getting control board and setting speed to zero before buoy search.");
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await?;
        println!("GOT ZERO SPEED SET");

        let mut can_see_buoy = true;

        while can_see_buoy {
            let camera_aquisition = self.context.get_front_camera_mat();
            let model_acquisition = buoy_model.detect(&camera_aquisition.await);
            match model_acquisition {
                Ok(acquisition_vec) => {
                    if acquisition_vec.len() == 0 {
                        can_see_buoy = false;
                    }
                    let detected_item = acquisition_vec
                        .iter()
                        .find(|&result| *result.class() == class_of_interest);
                    match detected_item {
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
                        None => can_see_buoy = false,
                    }
                }
                Err(_) => return Ok(()),
            }
        }
        // once we cannot see the buoy, we still want to continue forward, this implies we need to set the power to zero later.
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(self.forward_power, 0.0, 0.0, 0.0, self.target_depth)
            .await?;
        Ok(())
    }
}
