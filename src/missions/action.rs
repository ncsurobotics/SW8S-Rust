use async_trait::async_trait;
use core::fmt::Debug;
use std::marker::PhantomData;
use tokio::{join, try_join};

pub trait DrawGraph {}

/**
 * A trait for an action that can be executed.
 */
#[async_trait]
pub trait Action<Output: Send + Sync>: Send + Sync {
    async fn execute(self) -> Output;
}

/**
 * A trait that can be executed and modified at runtime.  
 */
pub trait ActionMod<Input: Send + Sync> {
    fn modify(&mut self, input: Input);
}

/**
 * An action that runs one of two actions depending on if its conditional reference is true or false.  
 */
#[derive(Debug)]
pub struct ActionConditional<U: Send + Sync, V: Action<bool>, W: Action<U>, X: Action<U>> {
    condition: V,
    true_branch: W,
    false_branch: X,
    _phantom_u: PhantomData<U>,
}

/**
 * Implementation for the ActionConditional struct.  
 */
impl<U: Send + Sync, V: Action<bool>, W: Action<U>, X: Action<U>> ActionConditional<U, V, W, X> {
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
impl<U: Send + Sync, V: Action<bool>, W: Action<U>, X: Action<U>> Action<U>
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
pub struct RaceAction<T: Action<bool>, U: Action<bool>> {
    first: T,
    second: U,
}

/**
 * Construct race action
 */
impl<T: Action<bool>, U: Action<bool>> RaceAction<T, U> {
    pub const fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
}

/**
 * Implement race logic where both actions are scheduled until one finishes.  
 */
#[async_trait]
impl<T: Action<bool>, U: Action<bool>> Action<bool> for RaceAction<T, U> {
    async fn execute(self) -> bool {
        self.first.execute().await || self.second.execute().await
    }
}

/**
 * Run two actions at once, and only exit when all actions have exited.
 */
#[derive(Debug)]
pub struct DualAction<T: Action<bool>, U: Action<bool>> {
    first: T,
    second: U,
}

/**
 * Constructor for the dual action
 */
impl<T: Action<bool>, U: Action<bool>> DualAction<T, U> {
    pub const fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
}

/**
 * Implement multiple logic where both actions are scheduled until both finish.  
 */
#[async_trait]
impl<T: Action<bool>, U: Action<bool>> Action<bool> for DualAction<T, U> {
    async fn execute(self) -> bool {
        self.first.execute().await && self.second.execute().await
    }
}

#[derive(Debug)]
pub struct ActionChain<T: Send + Sync, V: Action<T>, W: ActionMod<T>> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
}

impl<T: Send + Sync, V: Action<T>, W: ActionMod<T>> ActionChain<T, V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Send + Sync, U: Send + Sync, V: Action<T>, W: ActionMod<T> + Action<U>> Action<U>
    for ActionChain<T, V, W>
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
impl<T: Send + Sync, U: Send + Sync, V: Action<T>, W: Action<U>> Action<(T, U)>
    for ActionSequence<T, U, V, W>
{
    async fn execute(self) -> (T, U) {
        (self.first.execute().await, self.second.execute().await)
    }
}

#[derive(Debug)]
pub struct ActionParallel<T: Send + Sync, U: Send + Sync, V: Action<T>, W: Action<U>> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Send + Sync, U: Send + Sync, V: Action<T>, W: Action<U>> ActionParallel<T, U, V, W> {
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
        V: 'static + Action<T>,
        W: 'static + Action<U>,
    > Action<(T, U)> for ActionParallel<T, U, V, W>
{
    async fn execute(self) -> (T, U) {
        let fut1 = tokio::spawn(self.first.execute());
        let fut2 = tokio::spawn(self.second.execute());
        try_join!(fut1, fut2).unwrap()
    }
}

#[derive(Debug)]
pub struct ActionConcurrent<T: Send + Sync, U: Send + Sync, V: Action<T>, W: Action<U>> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Send + Sync, U: Send + Sync, V: Action<T>, W: Action<U>> ActionConcurrent<T, U, V, W> {
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
impl<T: Send + Sync, U: Send + Sync, V: Action<T>, W: Action<U>> Action<(T, U)>
    for ActionConcurrent<T, U, V, W>
{
    async fn execute(self) -> (T, U) {
        join!(self.first.execute(), self.second.execute())
    }
}
