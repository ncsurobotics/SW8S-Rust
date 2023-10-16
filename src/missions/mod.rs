use async_trait::async_trait;
use core::fmt::Debug;
use std::marker::PhantomData;
use tokio::{join, try_join, io::AsyncWriteExt};

use crate::comms::{control_board::ControlBoard, meb::MainElectronicsBoard};



/**
 * Trait that signifies a struct is an action dependency. 
 */
pub trait ActionContext {}

/**
 * Inherit this trait if you have a control board
 */
pub trait get_control_board<T: AsyncWriteExt + Unpin> {
    fn get_control_board(&self) -> &ControlBoard<T>;
}

/**
 * Inherit this trait if you have a MEB
 */
pub trait get_main_electronics_board {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard;
}

struct EmptyActionContext;

impl ActionContext for EmptyActionContext {}

struct FullActionContext<T: AsyncWriteExt + Unpin> {
    control_board: ControlBoard<T>,
    main_electronics_board: MainElectronicsBoard,
}
impl ActionContext for FullActionContext<tokio_serial::SerialStream> {}
impl<T: AsyncWriteExt + Unpin> FullActionContext<T> {
    const fn new(control_board: ControlBoard<T>, main_electronics_board: MainElectronicsBoard) -> Self {
        Self {
            control_board,
            main_electronics_board,
        }
    }
}
impl get_control_board<tokio_serial::SerialStream> for FullActionContext<tokio_serial::SerialStream> {
    fn get_control_board(&self) -> &ControlBoard<tokio_serial::SerialStream> {
        &self.control_board
    }
}
impl get_main_electronics_board for FullActionContext<tokio_serial::SerialStream> {
    fn get_main_electronics_board(&self) -> &MainElectronicsBoard {
        &self.main_electronics_board
    }
}

/**
 * A trait for an action that can be executed.
 */
#[async_trait]
pub trait Action<Output: Debug + Send + Sync>: Debug + Send + Sync + Sync {
    async fn execute(self) -> Output;
}

/**
 * A trait that can be executed and modified at runtime.  
 */
pub trait ActionMod<Input: ActionContext, Output: Debug + Send + Sync>:
    Action<Output>
{
    fn modify(&mut self, input: Input);
}

/**
 * An action that runs one of two actions depending on if its conditional reference is true or false.  
 */
#[derive(Debug)]
pub struct ActionConditional<U: ActionContext, V: Action<bool>, W: Action<U>, X: Action<U>> {
    condition: V,
    true_branch: W,
    false_branch: X,
    _phantom_u: PhantomData<U>,
}

/**
 * Implementation for the ActionConditional struct.  
 */
impl<U: Debug + Send + Sync, V: Action<bool>, W: Action<U>, X: Action<U>>
    ActionConditional<U, V, W, X>
{
    const fn new(condition: V, true_branch: W, false_branch: X) -> Self {
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
impl<U: Debug + Send + Sync, V: Action<bool>, W: Action<U>, X: Action<U>> Action<U>
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
    const fn new(first: T, second: U) -> Self {
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
    const fn new(first: T, second: U) -> Self {
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
pub struct ActionChain<
    T: Debug + Send + Sync,
    U: Debug + Send + Sync,
    V: Action<T>,
    W: ActionMod<T, U>,
> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: ActionMod<T, U>>
    ActionChain<T, U, V, W>
{
    const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: ActionMod<T, U>> Action<U>
    for ActionChain<T, U, V, W>
{
    async fn execute(mut self) -> U {
        self.second.modify(self.first.execute().await);
        self.second.execute().await
    }
}

#[derive(Debug)]
pub struct ActionSequence<
    T: Debug + Send + Sync,
    U: Debug + Send + Sync,
    V: Action<T>,
    W: Action<U>,
> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: Action<U>>
    ActionSequence<T, U, V, W>
{
    const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: Action<U>> Action<(T, U)>
    for ActionSequence<T, U, V, W>
{
    async fn execute(self) -> (T, U) {
        (self.first.execute().await, self.second.execute().await)
    }
}

#[derive(Debug)]
pub struct ActionParallel<
    T: Debug + Send + Sync,
    U: Debug + Send + Sync,
    V: Action<T>,
    W: Action<U>,
> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: Action<U>>
    ActionParallel<T, U, V, W>
{
    const fn new(first: V, second: W) -> Self {
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
        T: 'static + Debug + Send + Sync,
        U: 'static + Debug + Send + Sync,
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
pub struct ActionConcurrent<
    T: Debug + Send + Sync,
    U: Debug + Send + Sync,
    V: Action<T>,
    W: Action<U>,
> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: Action<U>>
    ActionConcurrent<T, U, V, W>
{
    const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<T: Debug + Send + Sync, U: Debug + Send + Sync, V: Action<T>, W: Action<U>> Action<(T, U)>
    for ActionConcurrent<T, U, V, W>
{
    async fn execute(self) -> (T, U) {
        join!(self.first.execute(), self.second.execute())
    }
}


