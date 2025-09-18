// Implementing <https://mb3hel.github.io/AUVControlBoard/user_guide/comm_protocol/>
use std::{error::Error, fmt::Display};

pub const START_BYTE: u8 = 253;
pub const END_BYTE: u8 = 254;
pub const ESCAPE_BYTE: u8 = 255;

/// See <https://github.com/ncsurobotics/SW8S-Java/blob/main/app/src/main/java/org/aquapackrobotics/sw8s/comms/CRC.java>
pub fn crc_itt16_false_bitmath(bytes: &[u8]) -> u16 {
    let mut crc = 0xFFFF;
    bytes.iter().for_each(|byte| {
        (0..8).for_each(|idx| {
            let bit: u8 = ((byte >> (7 - idx) & 1) == 1).into();
            let c15: u8 = ((crc >> 15 & 1) == 1).into();
            crc <<= 1;
            if (c15 ^ bit) != 0 {
                crc ^= 0x1021;
            }
        })
    });
    crc
}

/// Based on <https://github.com/ncsurobotics/SW8S-Java/blob/main/app/src/main/java/org/aquapackrobotics/sw8s/comms/CRC.java>
pub fn crc_itt16_false(bytes: &[u8]) -> u16 {
    let mut crc = 0xFFFF;
    bytes.iter().for_each(|byte| {
        (0..8).for_each(|idx| {
            let bit = (byte >> (7 - idx) & 1) == 1;
            let c15 = (crc >> 15 & 1) == 1;

            crc <<= 1;
            if c15 != bit {
                crc ^= 0x1021;
            }
        })
    });
    crc
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
        write!(f, "{self:?}")
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
