use async_trait::async_trait;

use super::action::{Action, ActionExec, ActionMod};

/// Development Action that does... nothing
///
/// Accepts any modification, for use as a development placeholder.
#[derive(Debug)]
pub struct NoOp {}

impl Default for NoOp {
    fn default() -> Self {
        Self::new()
    }
}

impl NoOp {
    pub fn new() -> Self {
        NoOp {}
    }
}

impl Action for NoOp {}

impl<T: Send + Sync> ActionMod<T> for NoOp {
    fn modify(&mut self, _input: &T) {}
}

#[async_trait]
impl ActionExec<()> for NoOp {
    async fn execute(&mut self) -> () {}
}

/// Always returns a true value
#[derive(Debug)]
pub struct AlwaysTrue {}

impl AlwaysTrue {
    pub fn new() -> Self {
        AlwaysTrue {}
    }
}
impl Default for AlwaysTrue {
    fn default() -> Self {
        Self::new()
    }
}

impl Action for AlwaysTrue {}

impl<T: Send + Sync> ActionMod<T> for AlwaysTrue {
    fn modify(&mut self, _input: &T) {}
}

#[async_trait]
impl ActionExec<anyhow::Result<()>> for AlwaysTrue {
    async fn execute(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ActionExec<Option<()>> for AlwaysTrue {
    async fn execute(&mut self) -> Option<()> {
        Some(())
    }
}

#[async_trait]
impl ActionExec<bool> for AlwaysTrue {
    async fn execute(&mut self) -> bool {
        true
    }
}

/// Unwraps Results and/or Options
#[derive(Debug)]
pub struct UnwrapAction<T> {
    action: T,
}

impl<T> UnwrapAction<T> {
    pub fn new(action: T) -> Self {
        UnwrapAction { action }
    }
}

impl<T> Action for UnwrapAction<T> {}

impl<T: ActionMod<U>, U: Send + Sync> ActionMod<U> for UnwrapAction<T> {
    fn modify(&mut self, input: &U) {
        self.action.modify(input)
    }
}

#[async_trait]
impl<T: ActionExec<anyhow::Result<U>>, U: Send + Sync> ActionExec<U> for UnwrapAction<T> {
    async fn execute(&mut self) -> U {
        self.action.execute().await.unwrap()
    }
}
