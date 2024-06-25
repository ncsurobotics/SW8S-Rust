use opencv::prelude::Mat;
use std::sync::Arc;
use std::sync::Mutex;

pub mod appsink;

#[allow(async_fn_in_trait)]
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

impl MatSource for SingleFrameSource {
    async fn get_mat(&self) -> Mat {
        self.inner.lock().unwrap().clone()
    }
}
