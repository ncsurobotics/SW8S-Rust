use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub speed: f32,
    pub forward_speed: f32,
    pub strafe_speed: f32,
    pub detections: u8,
    pub yaw_angle: f32,
    pub forward_duration: u64,
    pub yaw_wait: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            speed: 0.3,
            forward_speed: 0.3,
            strafe_speed: 0.3,
            detections: 10,
            yaw_angle: 15.0,
            forward_duration: 3,
            yaw_wait: 3,
        }
    }
}
