use async_trait::async_trait;
use core::fmt::Debug;
use std::marker::PhantomData;
use tokio::{join, try_join};

pub trait DrawGraph {}

/**
 * A trait for an action that can be executed.
 */
pub trait Action {}

/**
 * A trait for an action that can be executed.
 */
#[async_trait]
pub trait ActionExec<Output: Send + Sync>: Action + Send + Sync {
    async fn execute(self) -> Output;
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

impl<U, V: Action, W: Action, X: Action> Action for ActionConditional<U, V, W, X> {}

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
    async fn execute(self) -> U {
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
    async fn execute(self) -> bool {
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
    async fn execute(self) -> bool {
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
    async fn execute(mut self) -> U {
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
    async fn execute(self) -> (T, U) {
        (self.first.execute().await, self.second.execute().await)
    }
}

#[derive(Debug)]
pub struct ActionParallel<T: Send + Sync, U: Send + Sync, V: Action, W: Action> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Send + Sync, U: Send + Sync, V: Action, W: Action> Action for ActionParallel<T, U, V, W> {}

impl<T: Send + Sync, U: Send + Sync, V: Action, W: Action> ActionParallel<T, U, V, W> {
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
impl<
        T: 'static + Send + Sync,
        U: 'static + Send + Sync,
        V: 'static + ActionExec<T>,
        W: 'static + ActionExec<U>,
    > ActionExec<(T, U)> for ActionParallel<T, U, V, W>
{
    async fn execute(self) -> (T, U) {
        let fut1 = tokio::spawn(self.first.execute());
        let fut2 = tokio::spawn(self.second.execute());
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
    async fn execute(self) -> (T, U) {
        join!(self.first.execute(), self.second.execute())
    }
}
