use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub serial_port: PathBuf,
    pub serial_baud_rate: u32,
    pub bootloader: Bootloader,
    pub auto_transmit: AutoTransmit,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            serial_port: "/dev/ttyACM4".into(),
            serial_baud_rate: 115200,
            bootloader: Bootloader::default(),
            auto_transmit: AutoTransmit::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Bootloader {
    Skip = 0,
    Run = 1,
}

impl Default for Bootloader {
    fn default() -> Self {
        Self::Skip
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct AutoTransmit {
    pub mode: u8,
    pub gain_setting: Gain,
    pub transmit_duration: u16,
    pub sample_period: u16,
    pub transmit_frequency: u16,
    pub number_of_samples: u16,
    pub start_angle: u16,
    pub stop_angle: u16,
    pub num_steps: u8,
    pub delay: u8,
}

impl Default for AutoTransmit {
    fn default() -> Self {
        Self {
            mode: 1,
            gain_setting: Gain::default(),
            transmit_duration: 500,
            sample_period: 20000,
            transmit_frequency: 700,
            number_of_samples: 1000,
            start_angle: 0,
            stop_angle: 399,
            num_steps: 1,
            delay: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Gain {
    Low = 0,
    Normal = 1,
    High = 2,
}

impl Default for Gain {
    fn default() -> Self {
        Self::Normal
    }
}
