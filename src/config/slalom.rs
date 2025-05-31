use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub speed: f32,
    pub detections: u8,
    pub side: Side,
    pub centered_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            speed: 0.3,
            detections: 10,
            side: Side::Left,
            centered_threshold: 0.0,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Side {
    Right,
    Left,
}
