use std::fmt::Debug;
use std::marker::PhantomData;

use anyhow::{anyhow, bail};
use uuid::Uuid;

use crate::logln;

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

impl ActionExec<()> for NoOp {
    async fn execute(&mut self) {}
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

impl ActionExec<()> for Terminal {
    async fn execute(&mut self) {}
}

/// Nondisplaying action that resolves an ActionExec Type
#[derive(Debug)]
pub struct OutputType<T> {
    _phantom_t: PhantomData<T>,
}

impl<T> Default for OutputType<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OutputType<T> {
    pub fn new() -> Self {
        OutputType {
            _phantom_t: PhantomData,
        }
    }
}

impl<T> Action for OutputType<T> {
    fn dot_string(&self, _parent: &str) -> DotString {
        DotString {
            head_ids: vec![],
            tail_ids: vec![],
            body: "".to_string(),
        }
    }
}

impl<T: Send + Sync> ActionMod<T> for OutputType<T> {
    fn modify(&mut self, _input: &T) {}
}

impl<T: Send + Sync> ActionExec<()> for OutputType<T> {
    async fn execute(&mut self) {}
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

impl ActionExec<anyhow::Result<()>> for AlwaysTrue {
    async fn execute(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

impl ActionExec<Option<()>> for AlwaysTrue {
    async fn execute(&mut self) -> Option<()> {
        Some(())
    }
}

impl ActionExec<bool> for AlwaysTrue {
    async fn execute(&mut self) -> bool {
        true
    }
}

const TRUE: bool = true;

impl ActionExec<&'static bool> for AlwaysTrue {
    async fn execute(&mut self) -> &'static bool {
        &TRUE
    }
}

/// Always returns a false value
#[derive(Debug)]
pub struct AlwaysFalse {}

impl AlwaysFalse {
    pub fn new() -> Self {
        AlwaysFalse {}
    }
}
impl Default for AlwaysFalse {
    fn default() -> Self {
        Self::new()
    }
}

impl Action for AlwaysFalse {}

impl<T: Send + Sync> ActionMod<T> for AlwaysFalse {
    fn modify(&mut self, _input: &T) {}
}

impl ActionExec<anyhow::Result<()>> for AlwaysFalse {
    async fn execute(&mut self) -> anyhow::Result<()> {
        bail!("")
    }
}

impl ActionExec<Option<()>> for AlwaysFalse {
    async fn execute(&mut self) -> Option<()> {
        None
    }
}

impl ActionExec<bool> for AlwaysFalse {
    async fn execute(&mut self) -> bool {
        false
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
        logln!("Update true count: {}", self.count);
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
        logln!("Update true count: {}", self.count);
    }
}

impl ActionMod<bool> for CountTrue {
    fn modify(&mut self, input: &bool) {
        if *input {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
        logln!("Update true count: {}", self.count);
    }
}

impl ActionExec<anyhow::Result<()>> for CountTrue {
    async fn execute(&mut self) -> anyhow::Result<()> {
        logln!("Check true count: {} ? {}", self.count, self.target);
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
        logln!("Update false count: {}", self.count);
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
        logln!("Update false count: {}", self.count);
    }
}

impl ActionMod<bool> for CountFalse {
    fn modify(&mut self, input: &bool) {
        if !input {
            self.count += 1;
            if self.count > self.target {
                self.count = self.target;
            }
        } else {
            self.count = 0;
        }
        logln!("Update false count: {}", self.count);
    }
}

impl ActionExec<anyhow::Result<()>> for CountFalse {
    async fn execute(&mut self) -> anyhow::Result<()> {
        logln!("Check false count: {} ? {}", self.count, self.target);
        if self.count < self.target {
            Ok(())
        } else {
            Err(anyhow!("At count"))
        }
    }
}

#[derive(Debug)]
pub struct InOrderFail<T, U> {
    first: T,
    second: U,
    finished_first: bool,
}

impl<T, U> InOrderFail<T, U> {
    pub fn new(first: T, second: U) -> Self {
        Self {
            first,
            second,
            finished_first: false,
        }
    }
}

impl<T: Action, U: Action> Action for InOrderFail<T, U> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());
        let (order_head, order_tail) = (Uuid::new_v4(), Uuid::new_v4());

        let mut body_str = format!(
                "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = black;\n\"{}\" [label = \"{}\", shape = box, style = dashed];\n",
                Uuid::new_v4(),
                order_head,
                "In Order",
            );

        body_str.push_str(&(format!("\"{order_tail}\" [label = \"All Resolved\", shape = diamond, fontcolor = black, style = dashed];\n")));

        body_str.push_str(&(first_str.body + &second_str.body));
        first_str
            .head_ids
            .iter()
            .for_each(|id| body_str.push_str(&format!("\"{order_head}\" -> \"{id}\";\n")));
        first_str.tail_ids.iter().for_each(|tail_id| {
            second_str.head_ids.iter().for_each(|head_id| {
                body_str.push_str(&format!("\"{tail_id}\" -> \"{head_id}\";\n"))
            })
        });
        second_str.tail_ids.iter().for_each(|tail_id| {
            body_str.push_str(&format!("\"{tail_id}\" -> \"{order_tail}\";\n"))
        });

        body_str.push_str("}\n");

        DotString {
            head_ids: vec![order_head],
            tail_ids: vec![order_tail],
            body: body_str,
        }
    }
}

impl<T: ActionMod<V>, U: ActionMod<V>, V: Send + Sync> ActionMod<V> for InOrderFail<T, U> {
    fn modify(&mut self, input: &V) {
        self.first.modify(input);
        self.second.modify(input);
    }
}

impl<
        T: ActionExec<anyhow::Result<V>>,
        U: ActionExec<anyhow::Result<V>>,
        V: Send + Sync + Default,
    > ActionExec<anyhow::Result<V>> for InOrderFail<T, U>
{
    async fn execute(&mut self) -> anyhow::Result<V> {
        if !self.finished_first {
            let ret = self.first.execute().await;
            self.finished_first = ret.is_err();
            Ok(V::default())
        } else {
            self.second.execute().await
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

impl<T: Send + Sync + Debug, U: IntoIterator<Item = T> + Send + Sync + Clone> ActionMod<Option<U>>
    for ToVec<T>
{
    fn modify(&mut self, input: &Option<U>) {
        if let Some(input) = input {
            self.value = input.clone().into_iter().collect();
            logln!("VECTOR: {:#?}", self.value);
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

impl<T: Send + Sync + Clone> ActionExec<Vec<T>> for ToVec<T> {
    async fn execute(&mut self) -> Vec<T> {
        self.value.clone()
    }
}

/// Transform Option/Result wrapped vector to a vector
#[derive(Debug)]
pub struct IsSome<T> {
    value: Vec<T>,
}

impl<T> Action for IsSome<T> {}

impl<T> IsSome<T> {
    pub const fn new() -> Self {
        Self { value: vec![] }
    }
}

impl<T> Default for IsSome<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + Debug, U: IntoIterator<Item = T> + Send + Sync + Clone> ActionMod<Option<U>>
    for IsSome<T>
{
    fn modify(&mut self, input: &Option<U>) {
        if let Some(input) = input {
            self.value = input.clone().into_iter().collect();
            logln!("VECTOR: {:#?}", self.value);
        } else {
            self.value = vec![];
        }
    }
}

impl<T: Send + Sync, U: IntoIterator<Item = T> + Send + Sync + Clone> ActionMod<anyhow::Result<U>>
    for IsSome<T>
{
    fn modify(&mut self, input: &anyhow::Result<U>) {
        if let Ok(input) = input {
            self.value = input.clone().into_iter().collect();
        } else {
            self.value = vec![];
        }
    }
}

impl<T: Send + Sync + Clone> ActionExec<bool> for IsSome<T> {
    async fn execute(&mut self) -> bool {
        !self.value.is_empty()
    }
}

/// AlwaysBetter returns a true value
#[derive(Debug)]
pub struct AlwaysBetterTrue {}

impl AlwaysBetterTrue {
    pub fn new() -> Self {
        AlwaysBetterTrue {}
    }
}
impl Default for AlwaysBetterTrue {
    fn default() -> Self {
        Self::new()
    }
}

impl Action for AlwaysBetterTrue {}

impl<T: Send + Sync> ActionMod<T> for AlwaysBetterTrue {
    fn modify(&mut self, _input: &T) {}
}

impl ActionExec<anyhow::Result<()>> for AlwaysBetterTrue {
    async fn execute(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// AlwaysBetter returns a false value
#[derive(Debug)]
pub struct AlwaysBetterFalse {}

impl AlwaysBetterFalse {
    pub fn new() -> Self {
        AlwaysBetterFalse {}
    }
}
impl Default for AlwaysBetterFalse {
    fn default() -> Self {
        Self::new()
    }
}

impl Action for AlwaysBetterFalse {}

impl<T: Send + Sync> ActionMod<T> for AlwaysBetterFalse {
    fn modify(&mut self, _input: &T) {}
}

impl ActionExec<anyhow::Result<()>> for AlwaysBetterFalse {
    async fn execute(&mut self) -> anyhow::Result<()> {
        bail!("")
    }
}
