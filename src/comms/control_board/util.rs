use std::{error::Error, fmt::Display, ops::Deref};

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

#[derive(Debug)]
pub struct Thruster {
    inner: u8,
}

impl TryFrom<u8> for Thruster {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if (1..=8).contains(&value) {
            Ok(Thruster { inner: value })
        } else {
            bail!("{value} is not in range 1-8")
        }
    }
}

impl Deref for Thruster {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
