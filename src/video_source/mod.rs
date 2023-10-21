use async_trait::async_trait;
use opencv::prelude::Mat;

pub mod appsink;

#[async_trait]
trait MatSource {
    async fn get_mat(&self) -> Mat;
}
