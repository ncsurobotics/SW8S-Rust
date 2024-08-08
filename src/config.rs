use std::{
    fs::read_to_string,
    fs::write,
    ops::{Deref, DerefMut},
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigFile {
    pub control_board_path: String,
    pub control_board_backup_path: String,
    pub meb_path: String,
    pub front_cam: String,
    pub bottom_cam: String,
    pub standard_depth: f32,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self {
            control_board_path: "/dev/ttyACM0".to_string(),
            control_board_backup_path: "/dev/ttyACM3".to_string(),
            meb_path: "/dev/ttyACM2".to_string(),
            front_cam: "/dev/video1".to_string(),
            bottom_cam: "/dev/video0".to_string(),
            standard_depth: 1.0,
        }
    }
}

const CONFIG_FILE: &str = "config.toml";

#[derive(Debug)]
pub struct Configuration {
    inner: ConfigFile,
}

impl Default for Configuration {
    fn default() -> Self {
        let inner = if let Ok(config_string) = read_to_string(CONFIG_FILE) {
            match toml::from_str(&config_string) {
                Ok(x) => x,
                //Err(x) => panic!("Config file parsing: {:#?}", x),
                Err(_) => ConfigFile::default(),
            }
        } else {
            ConfigFile::default()
        };
        Self { inner }
    }
}

impl Drop for Configuration {
    fn drop(&mut self) {
        write(CONFIG_FILE, toml::to_string(&self.inner).unwrap()).unwrap();
    }
}

impl Deref for Configuration {
    type Target = ConfigFile;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Configuration {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
