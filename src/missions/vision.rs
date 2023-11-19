use std::marker::PhantomData;

use super::action::{Action, ActionExec};
use crate::video_source::MatSource;
use crate::vision::{Draw, Offset2D, RelPos, VisualDetector};
use anyhow::Result;
use async_trait::async_trait;
use num_traits::{FromPrimitive, Num};

#[cfg(feature = "logging")]
use opencv::{core::Vector, imgcodecs::imwrite};
#[cfg(feature = "logging")]
use std::fs::create_dir_all;
#[cfg(feature = "logging")]
use uuid::Uuid;

/// Runs a vision routine to obtain object position
///
/// The relative position is normalized to [-1, 1] on both axes
#[derive(Debug)]
pub struct VisionNormOffset<'a, T, U, V> {
    context: &'a T,
    model: U,
    _num: PhantomData<V>,
}

impl<'a, T, U, V> VisionNormOffset<'a, T, U, V> {
    pub const fn new(context: &'a T, model: U) -> Self {
        Self {
            context,
            model,
            _num: PhantomData,
        }
    }
}

impl<T, U, V> Action for VisionNormOffset<'_, T, U, V> {}

#[async_trait]
impl<T: MatSource, V: Num + FromPrimitive + Send + Sync, U: VisualDetector<V> + Send + Sync>
    ActionExec for VisionNormOffset<'_, T, U, V>
where
    U::Position: RelPos<Number = V> + Draw,
{
    type Output = Result<Offset2D<V>>;
    async fn execute(&mut self) -> Self::Output {
        #[cfg(feature = "logging")]
        {
            println!("Running detection...");
        }

        #[allow(unused_mut)]
        let mut mat = self.context.get_mat().await;
        let detections = self.model.detect(&mat);
        #[cfg(feature = "logging")]
        println!("Detect attempt: {}", detections.is_ok());
        let detections = detections?;
        #[cfg(feature = "logging")]
        {
            detections
                .iter()
                .for_each(|x| x.position().draw(&mut mat).unwrap());
            println!("Number of detects: {}", detections.len());
            create_dir_all("/tmp/detect").unwrap();
            imwrite(
                &("/tmp/detect/".to_string() + &Uuid::new_v4().to_string() + ".jpeg"),
                &mat,
                &Vector::default(),
            )
            .unwrap();
        }

        let positions: Vec<_> = detections
            .iter()
            .map(|detect| self.model.normalize(detect.position()))
            .map(|detect| detect.offset())
            .collect();

        let positions_len = positions.len();

        Ok(positions.into_iter().sum::<Offset2D<V>>() / positions_len)
    }
}
