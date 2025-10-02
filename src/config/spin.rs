use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub depth: f32,
    pub spin_speed: f32,
    pub num_spins: i32,
    pub hysteresis: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: -1.25,
            spin_speed: 1.0,
            num_spins: 2,
            hysteresis: 10.0,
        }
    }
}
