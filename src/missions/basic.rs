use super::{
    action::{Action, ActionConcurrent, ActionParallel, ActionSequence},
    meb::WaitArm,
    movement::Descend,
};

fn initial_descent<T: Send + Sync>(context: &T) -> impl Action + '_ {
    ActionSequence::<T, T, _, _>::new(
        ActionParallel::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -0.5)),
        ActionConcurrent::<T, T, _, _>::new(WaitArm::new(context), Descend::new(context, -1.0)),
    )
}
