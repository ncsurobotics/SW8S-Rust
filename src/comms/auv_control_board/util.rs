use std::{error::Error, fmt::Display};

/// Implementing https://mb3hel.github.io/AUVControlBoard/user_guide/comm_protocol/

pub const START_BYTE: u8 = 253;
pub const END_BYTE: u8 = 254;
pub const ESCAPE_BYTE: u8 = 255;

pub fn crc(_bytes: &[u8]) -> u16 {
    todo!()
}

#[derive(Debug)]
pub enum AcknowledgeErr {
    UnknownMsg,
    InvalidArguments,
    InvalidCommand,
    Reserved,
    Undefined(u8),
}

impl Display for AcknowledgeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for AcknowledgeErr {}

impl From<u8> for AcknowledgeErr {
    fn from(value: u8) -> Self {
        match value {
            1 => Self::UnknownMsg,
            2 => Self::InvalidArguments,
            3 => Self::InvalidCommand,
            255 => Self::Reserved,
            x => Self::Undefined(x),
        }
    }
}
