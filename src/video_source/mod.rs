use async_trait::async_trait;
use opencv::prelude::Mat;
use std::sync::Arc;
use std::sync::Mutex;

pub mod appsink;

#[async_trait]
pub trait MatSource: Send + Sync {
    async fn get_mat(&self) -> Mat;
}

#[derive(Debug)]
pub struct SingleFrameSource {
    inner: Arc<Mutex<Mat>>,
}

impl SingleFrameSource {
    pub fn new(frame: Mat) -> Self {
        Self {
            inner: Arc::new(Mutex::new(frame)),
        }
    }
}

#[async_trait]
impl MatSource for SingleFrameSource {
    async fn get_mat(&self) -> Mat {
        self.inner.lock().unwrap().clone()
    }
}
