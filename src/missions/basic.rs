use super::{
    action::{Action, ActionSequence},
    meb::WaitArm,
};
use anyhow::Result;

fn initial_descent<T, U, V>(context: &T) -> ActionSequence<T, T, WaitArm<T>, WaitArm<T>> {
    ActionSequence::new(WaitArm::new(context), WaitArm::new(context))
}
