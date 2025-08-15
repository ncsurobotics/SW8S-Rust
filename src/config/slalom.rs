use std::ops::RangeInclusive;

use super::Side;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub speed: f32,
    pub start_detections: u8,
    pub end_detections: u8,
    pub side: Side,
    pub centered_threshold: f32,
    pub dumb_strafe_secs: u64,
    pub init_duration: f32,
    pub strafe_duration: f32,
    pub traversal_duration: f32,
    pub yaw_adjustment: f32,
    pub yaw_speed: f32,
    pub area_bounds: RangeInclusive<f64>,
    pub correction_factor: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            speed: 0.3,
            start_detections: 10,
            end_detections: 10,
            side: Side::Left,
            centered_threshold: 0.0,
            dumb_strafe_secs: 2,
            init_duration: 1.0,
            strafe_duration: 2.0,
            traversal_duration: 6.0,
            yaw_adjustment: 15.0,
            yaw_speed: 0.2,
            area_bounds: 1000.0..=11000.0,
            correction_factor: 0.4,
        }
    }
}
