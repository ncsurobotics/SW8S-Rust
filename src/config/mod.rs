pub mod gate;
pub mod path_align;
pub mod slalom;

use std::{
    fs::{read_to_string, write},
    ops::{Deref, DerefMut},
    path::PathBuf,
};

use anyhow::Result;
use crossbeam::epoch::CompareAndSetOrdering;
use serde::{Deserialize, Serialize};

// Default values
const CONFIG_FILE: &str = "config.toml";
const CONTROL_BOARD_PATH: &str = "/dev/ttyACM0";
const CONTROL_BOARD_BACKUP_PATH: &str = "/dev/ttyACM3";
const MEB_PATH: &str = "/dev/ttyACM2";
const FRONT_CAM: &str = "/dev/video1";
const BOTTOM_CAM: &str = "/dev/video0";

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub control_board_path: String,
    pub control_board_backup_path: String,
    pub meb_path: String,
    pub front_cam_path: String,
    pub bottom_cam_path: String,
    pub missions: Missions,
}

impl Config {
    pub fn new() -> Result<Self> {
        let config_string = read_to_string(CONFIG_FILE)?;
        Ok(toml::from_str(&config_string)?)
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
            missions: Missions::default(),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Missions {
    pub gate: gate::Config,
    pub path_align: path_align::Config,
    pub slalom: slalom::Config,
}
