use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub speed: f32,
    pub true_count: u32,
    pub false_count: u32,
    pub side: Side,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            speed: 0.2,
            true_count: 4,
            false_count: 1,
            side: Side::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone)]
pub enum Side {
    Right,
    Left,
}

impl Default for Side {
    fn default() -> Self {
        Self::Right
    }
}
