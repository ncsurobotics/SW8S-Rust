use super::{
    action::{Action, ActionExec, ActionMod},
    action_context::GetControlBoard,
};

use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;


#[derive(Debug)]
struct DriveToBuoyVision<'a, T> {
    context: &'a T,
    buoy_model : Buoy,
    target_depth : f32,
}

impl<T> Action for DriveToBuoyVision<'_, T> {}

impl<T> ActionMod<f32> for Descend<'_, T> {
    fn modify(&mut self, depth: f32, buoy_model : Buoy) {
        self.target_depth = input;
        self.buoy_model = buoy_model;
    }
}

#[async_trait]
impl<T: GetControlBoard<WriteHalf<SerialStream>>> ActionExec for DriveToBuoyVision<'_, T> {
    type Output = Result<()>;
    async fn execute(&mut self) -> Self::Output {
        println!("Getting control board and setting speed to zero before buoy search.");
        self.context
            .get_control_board()
            .stability_2_speed_set_initial_yaw(0.0, 0.0, 0.0, 0.0, self.target_depth)
            .await?;
        println!("GOT SPEED SET");
        Ok(())
     }
}

