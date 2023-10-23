use anyhow::{anyhow, Result};
use async_trait::async_trait;
use opencv::prelude::Mat;
use opencv::videoio::VideoCapture;
use opencv::videoio::VideoCaptureAPIs;
use opencv::videoio::VideoCaptureTrait;
use std::sync::Arc;
use std::thread::spawn;
use std::{fs::create_dir, path::Path};
use tokio::sync::Mutex;

use super::MatSource;

#[derive(Debug)]
pub struct Camera {
    frame: Arc<Mutex<Option<Mat>>>,
}

impl Camera {
    pub fn new(
        camera_path: &str,
        camera_name: &str,
        filesink_dir: &Path,
        camera_dimensions: (u32, u32),
        rtsp: bool,
    ) -> Result<Self> {
        if !filesink_dir.is_dir() {
            create_dir(filesink_dir)?
        }

        let rtsp_string = "h264. ! queue ! h264parse config_interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! rtspclientsink location=rtsp://127.0.0.1:8554/".to_string()
                        + camera_name + " ";

        let capture_string =
            pipeline_head(camera_path, camera_dimensions.0, camera_dimensions.1, 30)
                + " ! jpegdec ! tee name=raw "
                + "raw. ! queue  ! videoconvert ! appsink "
                + "raw. ! queue  ! videoconvert ! "
                + &h264_enc_pipeline(2048000)
                + " ! tee name=h264 "
                + if rtsp { &rtsp_string } else { "" }
                + "h264. ! queue ! mpegtsmux ! filesink location=\""
                + filesink_dir
                    .to_str()
                    .ok_or(anyhow!("filesink_dir is not a string"))?
                + "/"
                + camera_name
                + "\" ";

        #[cfg(feature = "logging")]
        println!("Capture string: {capture_string}");
        let mut capture =
            VideoCapture::from_file(&capture_string, VideoCaptureAPIs::CAP_GSTREAMER as i32)?;

        let frame: Arc<Mutex<Option<Mat>>> = Arc::default();
        let frame_copy = frame.clone();

        spawn(move || loop {
            let mut mat = Mat::default();
            if capture.read(&mut mat).unwrap() {
                *frame_copy.blocking_lock() = Some(mat)
            }
        });

        Ok(Self { frame })
    }

    pub fn jetson_new(camera_path: &str, camera_name: &str, filesink_dir: &Path) -> Result<Self> {
        Camera::new(camera_path, camera_name, filesink_dir, (800, 600), true)
    }
}

#[async_trait]
impl MatSource for Camera {
    async fn get_mat(&self) -> Mat {
        loop {
            if let Some(mat) = self.frame.lock().await.take() {
                return mat;
            }
        }
    }
}

fn pipeline_head(device_name: &str, width: u32, height: u32, framerate: u32) -> String {
    #[cfg(target_os = "windows")]
    return format!("mfvideosrc device-index={device_name} ! image/jpeg, width={width}, height={height}, framerate={framerate}/1");

    #[cfg(not(target_os = "windows"))]
    return format!("v4l2src device={device_name} ! image/jpeg, width={width}, height={height}, framerate={framerate}/1");
}

fn h264_enc_pipeline(bitrate: u32) -> String {
    if Path::new("/etc/nv_tegra_release").exists() {
        format!(
            "omxh264enc bitrate={bitrate} control-rate=variable ! video/x-h264,profile=baseline"
        )
    } else {
        format!("x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate} ! video/x-h264,profile=baseline")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "requires an attached camera on a test system"]
    #[tokio::test]
    async fn single_camera() {
        // 640x360
        let output = Camera::new(
            "/dev/video0",
            "cam0",
            Path::new("/tmp/camera_test"),
            // Camera dependent parameter
            (640, 360),
            false,
        )
        .unwrap()
        .get_mat()
        .await;
        println!("{:?}", output);
    }
}
