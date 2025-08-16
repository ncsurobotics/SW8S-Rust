use super::Side;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub speed: f32,
    pub true_count: u32,
    pub false_count: u32,
    pub side: Side,
    pub yaw_speed: f32,
    pub strafe_speed: f32,
    pub init_duration: f32,
    pub strafe_duration: f32,
    pub traversal_duration: f32,
    pub yaw_adjustment: f32,
    pub correction_factor: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            speed: 0.2,
            true_count: 4,
            false_count: 1,
            side: Side::default(),
            yaw_speed: 0.2,
            strafe_speed: 0.2,
            correction_factor: 0.2,
            init_duration: 3.0,
            strafe_duration: 2.0,
            traversal_duration: 8.0,
            yaw_adjustment: 20.0,
        }
    }
}
