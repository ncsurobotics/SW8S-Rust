use std::{
    ops::{Deref, DerefMut},
    sync::{OnceLock, RwLock},
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigFile {
    pub control_board_path: String,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self {
            control_board_path: "".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct Configuration {
    inner: ConfigFile,
}

impl Configuration {
    fn default() -> Self {
        Self {
            inner: confy::load("sw8s").unwrap(),
        }
    }
}

impl Drop for Configuration {
    fn drop(&mut self) {
        confy::store("sw8s", &self.inner).unwrap();
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

static CONFIG_CELL: OnceLock<RwLock<Configuration>> = OnceLock::new();
pub fn configuration() -> &'static RwLock<Configuration> {
    CONFIG_CELL.get_or_init(|| RwLock::new(Configuration::default()))
}
