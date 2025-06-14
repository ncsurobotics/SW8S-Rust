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
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone)]
pub enum Side {
    Right,
    Left,
}
