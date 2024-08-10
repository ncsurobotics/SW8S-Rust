use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    comms::meb::{MainElectronicsBoard, MebCmd},
    logln,
};

use super::{
    action::{Action, ActionExec},
    action_context::GetMainElectronicsBoard,
};

#[derive(Debug)]
pub struct FireRightTorpedo<'a, T> {
    meb: &'a T,
}

impl<'a, T> FireRightTorpedo<'a, T> {
    pub fn new(meb: &'a T) -> Self {
        Self { meb }
    }
}

impl<T> Action for FireRightTorpedo<'_, T> {}

impl<T: GetMainElectronicsBoard> ActionExec<()> for FireRightTorpedo<'_, T> {
    async fn execute<'a>(&'a mut self) {
        let send_cmd = |meb: &'a MainElectronicsBoard<WriteHalf<SerialStream>>, cmd| async move {
            match meb.send_msg(cmd).await {
                Ok(()) => logln!("{:#?} success", cmd),
                Err(e) => logln!("{:#?} failure: {:#?}", cmd, e),
            };
        };

        let meb = self.meb.get_main_electronics_board();
        for _ in 0..3 {
            send_cmd(meb, MebCmd::T1Trig).await;
        }
    }
}

#[derive(Debug)]
pub struct FireLeftTorpedo<'a, T> {
    meb: &'a T,
}

impl<'a, T> FireLeftTorpedo<'a, T> {
    pub fn new(meb: &'a T) -> Self {
        Self { meb }
    }
}

impl<T> Action for FireLeftTorpedo<'_, T> {}

impl<T: GetMainElectronicsBoard> ActionExec<()> for FireLeftTorpedo<'_, T> {
    async fn execute<'a>(&'a mut self) {
        let send_cmd = |meb: &'a MainElectronicsBoard<WriteHalf<SerialStream>>, cmd| async move {
            match meb.send_msg(cmd).await {
                Ok(()) => logln!("{:#?} success", cmd),
                Err(e) => logln!("{:#?} failure: {:#?}", cmd, e),
            };
        };

        let meb = self.meb.get_main_electronics_board();
        for _ in 0..3 {
            send_cmd(meb, MebCmd::T2Trig).await;
        }
    }
}
