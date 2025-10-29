use std::os::raw::{c_long, c_ulong};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ByteMultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ColorRGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Duration {
    pub sec: c_long,
    pub nanosec: c_ulong,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Header {
    pub stamp: Time,
    pub frame_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Time {
    pub sec: c_long,
    pub nanosec: c_ulong,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Empty {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Float32MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Float64MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Int16MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<i16>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Int32MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Int64MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Int8MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<i8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiArrayDimension {
    pub label: String,
    pub size: u32,
    pub stride: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UInt16MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<u16>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UInt32MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UInt64MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UInt8MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiArrayLayout {
    pub dim: Vec<MultiArrayDimension>,
    pub data_offset: u32,
}
