use std::{
    fs::read_to_string,
    fs::write,
    ops::{Deref, DerefMut},
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigFile {
    pub control_board_path: String,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self {
            control_board_path: "/dev/ttyACM0".to_string(),
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
            toml::from_str(&config_string).unwrap()
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
