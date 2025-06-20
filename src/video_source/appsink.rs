use anyhow::{anyhow, Result};
use opencv::core::Size;
use opencv::mod_prelude::ToInputArray;
use opencv::prelude::Mat;
use opencv::videoio::VideoCaptureAPIs;
use opencv::videoio::{
    VideoCapture, VideoWriter, CAP_GSTREAMER, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
};
use opencv::videoio::{VideoCaptureTrait, VideoCaptureTraitConst, VideoWriterTrait};
use std::fs::create_dir_all;
use std::path::Path;
use std::sync;
use std::sync::Arc;
use std::thread::spawn;
use tokio::sync::Mutex;

use crate::logln;

use super::MatSource;

#[derive(Debug)]
pub struct Camera {
    frame: Arc<Mutex<Option<Mat>>>,
    #[cfg(feature = "annotated_streams")]
    output: Arc<sync::Mutex<VideoWriter>>,
}

impl Camera {
    pub fn new(
        camera_path: &str,
        camera_name: &str,
        filesink: &Path,
        camera_dimensions: (u32, u32),
        rtsp: bool,
    ) -> Result<Self> {
        if !filesink.is_dir() {
            create_dir_all(filesink)?
        }

        let rtsp_string = "h264. ! queue ! h264parse config_interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! rtspclientsink location=rtsp://127.0.0.1:8554/".to_string()
                        + camera_name + ".mp4 ";

        let capture_string =
            pipeline_head(camera_path, camera_dimensions.0, camera_dimensions.1, 30)
                + " ! jpegdec ! tee name=raw "
                + "raw. ! queue  ! videoconvert ! appsink "
                + "raw. ! queue  ! videoconvert ! "
                + &h264_enc_pipeline(2048000)
                + " ! tee name=h264 "
                + if rtsp { &rtsp_string } else { "" }
                + "h264. ! queue ! mpegtsmux ! filesink location=\""
                + filesink
                    .to_str()
                    .ok_or(anyhow!("filesink_dir is not a string"))?
                + "/"
                + camera_name
                + ".mp4\" ";

        #[cfg(feature = "annotated_streams")]
        let output_string = "appsrc ! videoconvert ! ".to_string()
            + &h264_enc_pipeline(2048000)
            + " ! mpegtsmux ! rtspclientsink location=rtspt://127.0.0.1:8554/"
            + camera_name
            + "_annotated.mp4 ";

        let frame: Arc<Mutex<Option<Mat>>> = Arc::default();
        let frame_copy = frame.clone();

        #[cfg(feature = "annotated_streams")]
        let output: Arc<sync::Mutex<VideoWriter>> =
            Arc::new(sync::Mutex::new(VideoWriter::default().unwrap()));
        #[cfg(feature = "annotated_streams")]
        let output_copy = output.clone();

        #[cfg(feature = "logging")]
        logln!("Capture string: {capture_string}");
        spawn(move || {
            let mut capture =
                VideoCapture::from_file(&capture_string, VideoCaptureAPIs::CAP_GSTREAMER as i32)
                    .unwrap();

            #[cfg(feature = "annotated_streams")]
            {
                let width = capture
                    .get(CAP_PROP_FRAME_WIDTH)
                    .expect("Failed to get capture frame width") as i32;
                let height = capture
                    .get(CAP_PROP_FRAME_HEIGHT)
                    .expect("Failed to get capture frame height")
                    as i32;
                let fps = capture
                    .get(CAP_PROP_FPS)
                    .expect("Failed to get capture FPS");

                *output_copy.lock().unwrap() = VideoWriter::new_with_backend_def(
                    &output_string,
                    CAP_GSTREAMER,
                    VideoWriter::fourcc('X', '2', '6', '4').unwrap(),
                    fps,
                    Size::new(width, height),
                )
                .unwrap();
            }
            loop {
                let mut mat = Mat::default();
                if capture.read(&mut mat).unwrap() {
                    *frame_copy.blocking_lock() = Some(mat)
                }
            }
        });

        Ok(Self {
            frame,
            #[cfg(feature = "annotated_streams")]
            output,
        })
    }

    pub fn jetson_new(camera_path: &str, camera_name: &str, filesink_dir: &Path) -> Result<Self> {
        Camera::new(camera_path, camera_name, filesink_dir, (640, 480), true)
    }

    #[cfg(feature = "annotated_streams")]
    pub fn push_annotated_frame(&self, image: &impl ToInputArray) {
        let writer = self.output.clone();
        let mut writer = writer.lock().unwrap();
        let _ = writer.write(image);
    }
}

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
        logln!("{:?}", output);
    }
}
