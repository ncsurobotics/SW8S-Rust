use tokio::io::WriteHalf;
use tokio_serial::SerialStream;
use crate::comms::meb::MainElectronicsBoard;
use crate::comms::meb::MebCmd;

use crate::{
    logln,
};

use super::{
    action::{Action, ActionExec},
    action_context::GetMainElectronicsBoard,
};

#[derive(Debug)]
pub struct ResetDropper<'a, T> {
    meb: &'a T,
}

impl<'a, T> ResetDropper<'a, T> {
    pub fn new(meb: &'a T) -> Self {
        Self { meb }
    }
}

impl<T> Action for ResetDropper<'_, T> {}

impl<T: GetMainElectronicsBoard> ActionExec<()> for ResetDropper<'_, T> {
    async fn execute<'a>(&'a mut self) {
        let send_cmd = |meb: &'a MainElectronicsBoard<WriteHalf<SerialStream>>, cmd| async move {
            match meb.send_dropper_msg(cmd).await {
                Ok(()) => logln!("{:#?} success", cmd),
                Err(e) => logln!("{:#?} failure: {:#?}", cmd, e),
            };
        };

        let meb = self.meb.get_main_electronics_board();
        send_cmd(meb, MebCmd::Reset).await; // Send the reset command
    }
}