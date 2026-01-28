pub mod bin;
pub mod coinflip;
pub mod gate;
pub mod octagon;
pub mod path_align;
pub mod slalom;
pub mod sonar;
pub mod spin;

use std::fs::read_to_string;

use crate::vision::Yuv;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::RangeInclusive;

pub const SHUTDOWN_TIMEOUT: u64 = 5;

// Default values
const CONFIG_FILE: &str = "config.toml";
const CONTROL_BOARD_PATH: &str = "/dev/ttyACM0";
const CONTROL_BOARD_BACKUP_PATH: &str = "/dev/ttyACM3";
const MEB_PATH: &str = "/dev/ttyACM2";
const FRONT_CAM: &str = "/dev/video0";
const BOTTOM_CAM: &str = "/dev/video1";

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub control_board_path: String,
    pub control_board_backup_path: String,
    pub meb_path: String,
    pub front_cam_path: String,
    pub bottom_cam_path: String,
    pub sonar: sonar::Config,
    pub missions: Missions,
    pub color_profile: String,
    pub color_profiles: HashMap<String, ColorProfile>,
    pub shark: Side,
    pub saw_fish: Side,
}

impl Config {
    pub fn new() -> Result<Self> {
        let config_string = read_to_string(CONFIG_FILE)?;
        Ok(toml::from_str(&config_string)?)
    }
}

impl Config {
    pub fn get_color_profile(&self) -> Option<&ColorProfile> {
        self.color_profiles.get(&self.color_profile)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            control_board_path: CONTROL_BOARD_PATH.to_string(),
            control_board_backup_path: CONTROL_BOARD_BACKUP_PATH.to_string(),
            meb_path: MEB_PATH.to_string(),
            front_cam_path: FRONT_CAM.to_string(),
            bottom_cam_path: BOTTOM_CAM.to_string(),
            sonar: sonar::Config::default(),
            missions: Missions::default(),
            color_profile: "".to_string(),
            color_profiles: HashMap::new(),
            shark: Side::default(),
            saw_fish: Side::default(),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Missions {
    pub gate: gate::Config,
    pub path_align: path_align::Config,
    pub slalom: slalom::Config,
    pub bin: bin::Config,
    pub octagon: octagon::Config,
    pub coinflip: coinflip::Config,
    pub spin: spin::Config,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ColorProfile {
    pub red: RangeInclusive<Yuv>,
    pub orange: RangeInclusive<Yuv>,
    pub yellow: RangeInclusive<Yuv>,
    pub purple: RangeInclusive<Yuv>,
    pub black: RangeInclusive<Yuv>,
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
