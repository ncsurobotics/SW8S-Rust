use std::{error::Error, fmt::Display};

use anyhow::bail;

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

/// See https://cdn-shop.adafruit.com/datasheets/BST_BNO055_DS000_12.pdf,
/// page 25
#[derive(Debug)]
pub enum BNO055AxisConfig {
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
}

impl TryFrom<u8> for BNO055AxisConfig {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::P0),
            1 => Ok(Self::P1),
            2 => Ok(Self::P2),
            3 => Ok(Self::P3),
            4 => Ok(Self::P4),
            5 => Ok(Self::P5),
            6 => Ok(Self::P6),
            7 => Ok(Self::P7),
            x => bail!("{x} is >= 8, invalid axis config value"),
        }
    }
}

impl From<BNO055AxisConfig> for u8 {
    fn from(val: BNO055AxisConfig) -> Self {
        match val {
            BNO055AxisConfig::P0 => 0,
            BNO055AxisConfig::P1 => 1,
            BNO055AxisConfig::P2 => 2,
            BNO055AxisConfig::P3 => 3,
            BNO055AxisConfig::P4 => 4,
            BNO055AxisConfig::P5 => 5,
            BNO055AxisConfig::P6 => 6,
            BNO055AxisConfig::P7 => 7,
        }
    }
}
