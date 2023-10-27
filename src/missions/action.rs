use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use std::{marker::PhantomData, sync::Arc};
use tokio::{join, sync::Mutex, try_join};
use uuid::Uuid;

use super::graph::{stripped_type, DotString};

/**
 * A trait for an action that can be executed.
 */
pub trait Action {
    /// Represent this node in dot (graphviz) notation
    fn dot_string(&self) -> DotString {
        let id = Uuid::new_v4();
        DotString {
            head_id: id,
            tail_ids: vec![id],
            body: format!("\"{}\" [label = \"{}\"]", id, stripped_type::<Self>()),
        }
    }
}

/**
 * A trait for an action that can be executed.
 */
#[async_trait]
pub trait ActionExec<Output: Send + Sync>: Action + Send + Sync {
    async fn execute(&mut self) -> Output;
}

/**
 * A trait that can be executed and modified at runtime.  
 */
pub trait ActionMod<Input: Send + Sync>: Action {
    fn modify(&mut self, input: Input);
}

/**
 * An action that runs one of two actions depending on if its conditional reference is true or false.  
 */
#[derive(Debug)]
pub struct ActionConditional<U, V: Action, W: Action, X: Action> {
    condition: V,
    true_branch: W,
    false_branch: X,
    _phantom_u: PhantomData<U>,
}

impl<U, V: Action, W: Action, X: Action> Action for ActionConditional<U, V, W, X> {
    fn dot_string(&self) -> DotString {
        let true_str = self.true_branch.dot_string();
        let false_str = self.true_branch.dot_string();
        let condition_str = self.condition.dot_string();

        let combined_str = true_str.body
            + "\n"
            + &false_str.body
            + "\n"
            + &condition_str.body
            + "\n"
            + &format!("\"{}\" [shape = diamond];\n", condition_str.tail_ids[0])
            + &format!(
                "\"{}\" -> \"{}\" [label = \"True\"];\n",
                condition_str.tail_ids[0], true_str.head_id,
            )
            + &format!(
                "\"{}\" -> \"{}\" [label = \"False\"];",
                condition_str.tail_ids[0], false_str.head_id,
            );
        DotString {
            head_id: condition_str.head_id,
            tail_ids: vec![true_str.head_id, false_str.head_id],
            body: combined_str,
        }
    }
}

/**
 * Implementation for the ActionConditional struct.  
 */
impl<U, V: Action, W: Action, X: Action> ActionConditional<U, V, W, X> {
    pub const fn new(condition: V, true_branch: W, false_branch: X) -> Self {
        Self {
            condition,
            true_branch,
            false_branch,
            _phantom_u: PhantomData,
        }
    }
}

/**
 * Implement the conditional logic for the ActionConditional action.
 */
#[async_trait]
impl<U: Send + Sync, V: ActionExec<bool>, W: ActionExec<U>, X: ActionExec<U>> ActionExec<U>
    for ActionConditional<U, V, W, X>
{
    async fn execute(&mut self) -> U {
        if self.condition.execute().await {
            self.true_branch.execute().await
        } else {
            self.false_branch.execute().await
        }
    }
}

#[derive(Debug)]
/**
 * Action that runs two actions at the same time and exits both when one exits
 */
pub struct RaceAction<T: Action, U: Action> {
    first: T,
    second: U,
}

impl<T: Action, U: Action> Action for RaceAction<T, U> {}

/**
 * Construct race action
 */
impl<T: Action, U: Action> RaceAction<T, U> {
    pub const fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
}

/**
 * Implement race logic where both actions are scheduled until one finishes.  
 */
#[async_trait]
impl<T: ActionExec<bool>, U: ActionExec<bool>> ActionExec<bool> for RaceAction<T, U> {
    async fn execute(&mut self) -> bool {
        self.first.execute().await || self.second.execute().await
    }
}

/**
 * Run two actions at once, and only exit when all actions have exited.
 */
#[derive(Debug)]
pub struct DualAction<T: Action, U: Action> {
    first: T,
    second: U,
}

impl<T: Action, U: Action> Action for DualAction<T, U> {}

/**
 * Constructor for the dual action
 */
impl<T: Action, U: Action> DualAction<T, U> {
    pub const fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
}

/**
 * Implement multiple logic where both actions are scheduled until both finish.  
 */
#[async_trait]
impl<T: ActionExec<bool>, U: ActionExec<bool>> ActionExec<bool> for DualAction<T, U> {
    async fn execute(&mut self) -> bool {
        self.first.execute().await && self.second.execute().await
    }
}

#[derive(Debug)]
pub struct ActionChain<T: Sync + Send, V: Action, W: ActionMod<T>> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
}

impl<T: Sync + Send, V: Action, W: ActionMod<T>> Action for ActionChain<T, V, W> {}

impl<T: Sync + Send, V: Action, W: ActionMod<T>> ActionChain<T, V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Send + Sync, U: Send + Sync, V: ActionExec<T>, W: ActionMod<T> + ActionExec<U>>
    ActionExec<U> for ActionChain<T, V, W>
{
    async fn execute(&mut self) -> U {
        self.second.modify(self.first.execute().await);
        self.second.execute().await
    }
}

#[derive(Debug)]
pub struct ActionSequence<T, U, V, W> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T, U, V: Action, W: Action> Action for ActionSequence<T, U, V, W> {}

impl<T, U, V, W> ActionSequence<T, U, V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Send + Sync, U: Send + Sync, V: ActionExec<T>, W: ActionExec<U>> ActionExec<(T, U)>
    for ActionSequence<T, U, V, W>
{
    async fn execute(&mut self) -> (T, U) {
        (self.first.execute().await, self.second.execute().await)
    }
}

#[derive(Debug)]
pub struct ActionParallel<T: Send + Sync, U: Send + Sync, V: Action, W: Action> {
    first: Arc<Mutex<V>>,
    second: Arc<Mutex<W>>,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Send + Sync, U: Send + Sync, V: Action, W: Action> Action for ActionParallel<T, U, V, W> {}

impl<T: Send + Sync, U: Send + Sync, V: Action, W: Action> ActionParallel<T, U, V, W> {
    pub fn new(first: V, second: W) -> Self {
        Self {
            first: Arc::from(Mutex::from(first)),
            second: Arc::from(Mutex::from(second)),
            _phantom_t: PhantomData,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<
        T: 'static + Send + Sync,
        U: 'static + Send + Sync,
        V: 'static + ActionExec<T>,
        W: 'static + ActionExec<U>,
    > ActionExec<(T, U)> for ActionParallel<T, U, V, W>
{
    async fn execute(&mut self) -> (T, U) {
        let first = self.first.clone();
        let second = self.second.clone();
        let fut1 = tokio::spawn(async move { first.lock().await.execute().await });
        let fut2 = tokio::spawn(async move { second.lock().await.execute().await });
        try_join!(fut1, fut2).unwrap()
    }
}

#[derive(Debug)]
pub struct ActionConcurrent<T, U, V: Action, W: Action> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T, U, V: Action, W: Action> Action for ActionConcurrent<T, U, V, W> {}

impl<T, U, V: Action, W: Action> ActionConcurrent<T, U, V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Send + Sync, U: Send + Sync, V: ActionExec<T>, W: ActionExec<U>> ActionExec<(T, U)>
    for ActionConcurrent<T, U, V, W>
{
    async fn execute(&mut self) -> (T, U) {
        join!(self.first.execute(), self.second.execute())
    }
}

/**
 * An action that tries `count` times for a success
 */
#[derive(Debug)]
pub struct ActionUntil<T: Action> {
    action: T,
    limit: u32,
}

impl<T: Action> Action for ActionUntil<T> {}

impl<T: Action> ActionUntil<T> {
    pub const fn new(action: T, limit: u32) -> Self {
        Self { action, limit }
    }
}

#[async_trait]
impl<U: Send + Sync, T: ActionExec<Result<U>>> ActionExec<Result<U>> for ActionUntil<T> {
    async fn execute(&mut self) -> Result<U> {
        let mut count = 1;
        let mut result = self.action.execute().await;
        while result.is_err() && count < self.limit {
            result = self.action.execute().await;
            count += 1;
        }
        result
    }
}

/**
 * An action that runs while true
 */
#[derive(Debug)]
pub struct ActionWhile<T: Action> {
    action: T,
}

impl<T: Action> Action for ActionWhile<T> {}

/**
 * Implementation for the ActionWhile struct.  
 */
impl<T: Action> ActionWhile<T> {
    pub const fn new(action: T) -> Self {
        Self { action }
    }
}

#[async_trait]
impl<T: ActionExec<bool>> ActionExec<()> for ActionWhile<T> {
    async fn execute(&mut self) -> () {
        while self.action.execute().await {}
    }
}

#[async_trait]
impl<U: Send + Sync, T: ActionExec<Result<U>>> ActionExec<Result<U>> for ActionWhile<T> {
    async fn execute(&mut self) -> Result<U> {
        let mut result = self.action.execute().await;
        while result.is_ok() {
            result = self.action.execute().await;
        }
        result
    }
}
