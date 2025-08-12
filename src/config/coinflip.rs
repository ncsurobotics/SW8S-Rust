use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub angle_correction: f32,
    pub true_count: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            angle_correction: 15.0,
            true_count: 4,
        }
    }
}
