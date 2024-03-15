use anyhow::anyhow;
use async_trait::async_trait;
use uuid::Uuid;

use super::{
    action::{Action, ActionExec, ActionMod},
    graph::{stripped_fn, stripped_type, DotString},
};

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

/// [`NoOp`], but does not display on graph
#[derive(Debug)]
pub struct Terminal {}

impl Default for Terminal {
    fn default() -> Self {
        Self::new()
    }
}

impl Terminal {
    pub fn new() -> Self {
        Terminal {}
    }
}

impl Action for Terminal {
    fn dot_string(&self, _parent: &str) -> DotString {
        DotString {
            head_ids: vec![],
            tail_ids: vec![],
            body: "".to_string(),
        }
    }
}

impl<T: Send + Sync> ActionMod<T> for Terminal {
    fn modify(&mut self, _input: &T) {}
}

#[async_trait]
impl ActionExec<()> for Terminal {
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

#[derive(Debug)]
pub struct CountTrue {
    target: u32,
    count: u32,
}

impl CountTrue {
    pub fn new(target: u32) -> Self {
        CountTrue { target, count: 0 }
    }
}

impl Action for CountTrue {
    fn dot_string(&self, _parent: &str) -> DotString {
        let id = Uuid::new_v4();
        DotString {
            head_ids: vec![id],
            tail_ids: vec![id],
            body: format!(
                "\"{}\" [label = \"Consecutive True < {}\", margin = 0];\n",
                id, self.target
            ),
        }
    }
}

impl<T: Send + Sync> ActionMod<anyhow::Result<T>> for CountTrue {
    fn modify(&mut self, input: &anyhow::Result<T>) {
        if input.is_ok() {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
    }
}

impl<T: Send + Sync> ActionMod<Option<T>> for CountTrue {
    fn modify(&mut self, input: &Option<T>) {
        if input.is_some() {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
    }
}

#[async_trait]
impl ActionExec<anyhow::Result<()>> for CountTrue {
    async fn execute(&mut self) -> anyhow::Result<()> {
        if self.count < self.target {
            Ok(())
        } else {
            Err(anyhow!("At count"))
        }
    }
}

#[derive(Debug)]
pub struct CountFalse {
    target: u32,
    count: u32,
}

impl CountFalse {
    pub fn new(target: u32) -> Self {
        CountFalse { target, count: 0 }
    }
}

impl Action for CountFalse {
    fn dot_string(&self, _parent: &str) -> DotString {
        let id = Uuid::new_v4();
        DotString {
            head_ids: vec![id],
            tail_ids: vec![id],
            body: format!(
                "\"{}\" [label = \"Consecutive False < {}\", margin = 0];\n",
                id, self.target
            ),
        }
    }
}

impl<T: Send + Sync> ActionMod<anyhow::Result<T>> for CountFalse {
    fn modify(&mut self, input: &anyhow::Result<T>) {
        if input.is_err() {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
    }
}

impl<T: Send + Sync> ActionMod<Option<T>> for CountFalse {
    fn modify(&mut self, input: &Option<T>) {
        if input.is_none() {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
    }
}

#[async_trait]
impl ActionExec<anyhow::Result<()>> for CountFalse {
    async fn execute(&mut self) -> anyhow::Result<()> {
        if self.count < self.target {
            Ok(())
        } else {
            Err(anyhow!("At count"))
        }
    }
}

/// Transform `value` with `transform_function`.
///
/// Generic action for secondary transformations
#[derive(Debug)]
pub struct Transform<T, U, V: Fn(T) -> U> {
    value: T,
    transform_function: V,
}

impl<T, U, V: Fn(T) -> U> Action for Transform<T, U, V> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let id = Uuid::new_v4();
        DotString {
            head_ids: vec![id],
            tail_ids: vec![id],
            body: format!(
                "\"{}\" [label = \"{}\", margin = 0];\n",
                id,
                stripped_fn::<V>()
            ),
        }
    }
}

impl<T, U, V: Fn(T) -> U> Transform<T, U, V> {
    pub const fn new(value: T, transform_function: V) -> Self {
        Self {
            value,
            transform_function,
        }
    }
}

impl<T: Default, U, V: Fn(T) -> U> Transform<T, U, V> {
    pub fn new_default(transform_function: V) -> Self {
        Self::new(T::default(), transform_function)
    }
}

impl<T: Send + Sync + Clone, U, V: Fn(T) -> U> ActionMod<T> for Transform<T, U, V> {
    fn modify(&mut self, input: &T) {
        self.value = input.clone();
    }
}

#[async_trait]
impl<T: Send + Sync + Clone, U: Send + Sync, V: Fn(T) -> U + Send + Sync> ActionExec<U>
    for Transform<T, U, V>
{
    async fn execute(&mut self) -> U {
        (self.transform_function)(self.value.clone())
    }
}

/// Transform Option/Result wrapped vector to a vector
#[derive(Debug)]
pub struct ToVec<T> {
    value: Vec<T>,
}

impl<T> Action for ToVec<T> {}

impl<T> ToVec<T> {
    pub const fn new() -> Self {
        Self { value: vec![] }
    }
}

impl<T> Default for ToVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync, U: IntoIterator<Item = T> + Send + Sync + Clone> ActionMod<Option<U>>
    for ToVec<T>
{
    fn modify(&mut self, input: &Option<U>) {
        if let Some(input) = input {
            self.value = input.clone().into_iter().collect();
        } else {
            self.value = vec![];
        }
    }
}

impl<T: Send + Sync, U: IntoIterator<Item = T> + Send + Sync + Clone> ActionMod<anyhow::Result<U>>
    for ToVec<T>
{
    fn modify(&mut self, input: &anyhow::Result<U>) {
        if let Ok(input) = input {
            self.value = input.clone().into_iter().collect();
        } else {
            self.value = vec![];
        }
    }
}

#[async_trait]
impl<T: Send + Sync + Clone> ActionExec<Vec<T>> for ToVec<T> {
    async fn execute(&mut self) -> Vec<T> {
        self.value.clone()
    }
}
