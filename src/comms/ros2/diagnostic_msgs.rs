use crate::comms::ros2::std_msgs::Header;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DiagnosticArray {
    pub header: Header,
    pub status: Vec<DiagnosticStatus>,
}

pub const OK: u8 = 0;
pub const WARN: u8 = 1;
pub const ERROR: u8 = 2;
pub const STALE: u8 = 3;

#[derive(Debug, Serialize, Deserialize)]
pub struct DiagnosticStatus {
    pub level: u8,
    pub name: String,
    pub message: String,
    pub hardware_id: String,
    pub values: Vec<KeyValue>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}
