use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use std::{marker::PhantomData, sync::Arc, thread};
use tokio::{join, runtime::Handle, sync::Mutex};
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
            head_ids: vec![id],
            tail_ids: vec![id],
            body: format!("\"{}\" [label = \"{}\"];\n", id, stripped_type::<Self>()),
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
        let false_str = self.false_branch.dot_string();
        let condition_str = self.condition.dot_string();

        let mut combined_str = true_str.body + &false_str.body + &condition_str.body;
        for tail_id in condition_str.tail_ids {
            combined_str.push_str(&format!("\"{}\" [shape = diamond];\n", tail_id));
            for head_id in &true_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{}\" -> \"{}\" [label = \"True\"];\n",
                    tail_id, head_id,
                ));
            }
            for head_id in &false_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{}\" -> \"{}\" [label = \"False\"];\n",
                    tail_id, head_id,
                ));
            }
        }
        DotString {
            head_ids: condition_str.head_ids,
            tail_ids: vec![true_str.head_ids, false_str.head_ids]
                .into_iter()
                .flatten()
                .collect(),
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

impl<T: Action, U: Action> Action for RaceAction<T, U> {
    fn dot_string(&self) -> DotString {
        let first_str = self.first.dot_string();
        let second_str = self.second.dot_string();
        let race_id = Uuid::new_v4();

        let mut body_str = format!(
            "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = red;\n\"{}\" [label = \"Race\", shape = box, fontcolor = red, style = dashed];\n",
            Uuid::new_v4(),
            race_id
        ) + &first_str.body
            + &second_str.body;

        vec![first_str.head_ids, second_str.head_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", race_id, id)));
        body_str.push_str("}\n");

        DotString {
            head_ids: vec![race_id],
            tail_ids: vec![first_str.tail_ids, second_str.tail_ids]
                .into_iter()
                .flatten()
                .collect(),
            body: body_str,
        }
    }
}

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
impl<V: Sync + Send, T: ActionExec<V>, U: ActionExec<V>> ActionExec<V> for RaceAction<T, U> {
    async fn execute(&mut self) -> V {
        tokio::select! {
            res = self.first.execute() => res,
            res = self.second.execute() => res
        }
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

impl<T: Action, U: Action> Action for DualAction<T, U> {
    fn dot_string(&self) -> DotString {
        let first_str = self.first.dot_string();
        let second_str = self.second.dot_string();
        let (dual_head, dual_tail) = (Uuid::new_v4(), Uuid::new_v4());

        let mut body_str = format!(
            "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = blue;\n\"{}\" [label = \"Dual\", shape = box, fontcolor = blue, style = dashed];\n",
            Uuid::new_v4(),
            dual_head
        ) + &format!("{}\" [label = \"Collect\", shape = box, fontcolor = blue, style = dashed];\n", dual_tail) +
            &first_str.body
            + &second_str.body;

        vec![first_str.head_ids, second_str.head_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", dual_head, id)));
        vec![first_str.tail_ids, second_str.tail_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", id, dual_tail)));
        body_str.push_str("}\n");

        DotString {
            head_ids: vec![dual_head],
            tail_ids: vec![dual_tail],
            body: body_str,
        }
    }
}

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
impl<V: Send + Sync, T: ActionExec<V>, U: ActionExec<V>> ActionExec<(V, V)> for DualAction<T, U> {
    async fn execute(&mut self) -> (V, V) {
        tokio::join!(self.first.execute(), self.second.execute())
    }
}

#[derive(Debug)]
pub struct ActionChain<T: Sync + Send, V: Action, W: Action> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
}

impl<T: Sync + Send, V: Action, W: Action> Action for ActionChain<T, V, W> {
    fn dot_string(&self) -> DotString {
        let first_str = self.first.dot_string();
        let second_str = self.second.dot_string();

        let mut body_str = first_str.body + &second_str.body;
        for tail in &first_str.tail_ids {
            for head in &second_str.head_ids {
                body_str.push_str(&format!(
                    "\"{}\" -> \"{}\" [color = purple, textcolor = purple, label = \"Pass Data\"];\n",
                    tail, head
                ))
            }
        }

        DotString {
            head_ids: first_str.head_ids,
            tail_ids: second_str.tail_ids,
            body: body_str,
        }
    }
}

impl<T: Sync + Send, V: Action, W: Action> ActionChain<T, V, W> {
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

impl<T, U, V: Action, W: Action> Action for ActionSequence<T, U, V, W> {
    fn dot_string(&self) -> DotString {
        let first_str = self.first.dot_string();
        let second_str = self.second.dot_string();

        let mut body_str = first_str.body + &second_str.body;
        for tail in &first_str.tail_ids {
            for head in &second_str.head_ids {
                body_str.push_str(&format!("\"{}\" -> \"{}\";\n", tail, head))
            }
        }

        DotString {
            head_ids: first_str.head_ids,
            tail_ids: second_str.tail_ids,
            body: body_str,
        }
    }
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
impl<
        T: Send + Sync,
        U: Send + Sync,
        X: Send + Sync,
        Y: Send + Sync,
        V: ActionExec<Y>,
        W: ActionExec<X>,
    > ActionExec<(Y, X)> for ActionSequence<T, U, V, W>
{
    async fn execute(&mut self) -> (Y, X) {
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

impl<T: Send + Sync, U: Send + Sync, V: Action, W: Action> Action for ActionParallel<T, U, V, W> {
    fn dot_string(&self) -> DotString {
        let first_str = self.first.blocking_lock().dot_string();
        let second_str = self.second.blocking_lock().dot_string();
        let (par_head, par_tail) = (Uuid::new_v4(), Uuid::new_v4());

        let mut body_str = format!(
            "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = blue;\n\"{}\" [label = \"Parallel\", shape = box, fontcolor = blue, style = dashed];\n",
            Uuid::new_v4(),
            par_head
        ) + &format!("{}\" [label = \"Collect\", shape = box, fontcolor = blue, style = dashed];\n", par_tail) +
            &first_str.body
            + &second_str.body;

        vec![first_str.head_ids, second_str.head_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", par_head, id)));
        vec![first_str.tail_ids, second_str.tail_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", id, par_tail)));
        body_str.push_str("}\n");

        DotString {
            head_ids: vec![par_head],
            tail_ids: vec![par_tail],
            body: body_str,
        }
    }
}

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
        Y: 'static + Send + Sync,
        X: 'static + Send + Sync,
        V: 'static + ActionExec<Y>,
        W: 'static + ActionExec<X>,
    > ActionExec<(Y, X)> for ActionParallel<T, U, V, W>
{
    async fn execute(&mut self) -> (Y, X) {
        let first = self.first.clone();
        let second = self.second.clone();
        let handle1 = Handle::current();
        let handle2 = Handle::current();

        // https://docs.rs/tokio/1.33.0/tokio/runtime/struct.Handle.html#method.block_on
        let fut1 = thread::spawn(move || {
            handle1.block_on(async move { first.lock().await.execute().await })
        });
        let fut2 = thread::spawn(move || {
            handle2.block_on(async move { second.lock().await.execute().await })
        });
        (fut1.join().unwrap(), fut2.join().unwrap())
    }
}

#[derive(Debug)]
pub struct ActionConcurrent<T, U, V: Action, W: Action> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
    _phantom_u: PhantomData<U>,
}

impl<T, U, V: Action, W: Action> Action for ActionConcurrent<T, U, V, W> {
    fn dot_string(&self) -> DotString {
        let first_str = self.first.dot_string();
        let second_str = self.second.dot_string();
        let (concurrent_head, concurrent_tail) = (Uuid::new_v4(), Uuid::new_v4());

        let mut body_str = format!(
            "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = blue;\n\"{}\" [label = \"Concurrent\", shape = box, fontcolor = blue, style = dashed];\n",
            Uuid::new_v4(),
            concurrent_head
        ) + &format!("{}\" [label = \"Collect\", shape = box, fontcolor = blue, style = dashed];\n", concurrent_tail) +
            &first_str.body
            + &second_str.body;

        vec![first_str.head_ids, second_str.head_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", concurrent_head, id)));
        vec![first_str.tail_ids, second_str.tail_ids]
            .into_iter()
            .flatten()
            .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", id, concurrent_tail)));
        body_str.push_str("}\n");

        DotString {
            head_ids: vec![concurrent_head],
            tail_ids: vec![concurrent_tail],
            body: body_str,
        }
    }
}

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
impl<
        T: Send + Sync,
        U: Send + Sync,
        X: Send + Sync,
        Y: Send + Sync,
        V: ActionExec<Y>,
        W: ActionExec<X>,
    > ActionExec<(Y, X)> for ActionConcurrent<T, U, V, W>
{
    async fn execute(&mut self) -> (Y, X) {
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

impl<T: Action> Action for ActionUntil<T> {
    fn dot_string(&self) -> DotString {
        let action_str = self.action.dot_string();

        let mut body_str = action_str.body;
        for head in &action_str.head_ids {
            body_str.push_str(&format!("\"{}\" [shape = diamond];\n", head));
            for tail in &action_str.tail_ids {
                body_str.push_str(&format!(
                    "\"{}\":sw -> \"{}\":nw [label = \"Fail Within Count\"];\n",
                    tail, head
                ))
            }
        }

        DotString {
            head_ids: action_str.head_ids,
            tail_ids: action_str.tail_ids,
            body: body_str,
        }
    }
}

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

impl<T: Action> Action for ActionWhile<T> {
    fn dot_string(&self) -> DotString {
        let action_str = self.action.dot_string();

        let mut body_str = action_str.body;
        for head in &action_str.head_ids {
            body_str.push_str(&format!("\"{}\" [shape = diamond];\n", head));
            for tail in &action_str.tail_ids {
                body_str.push_str(&format!(
                    "\"{}\":sw -> \"{}\":nw [label = \"True\"];\n",
                    tail, head
                ))
            }
        }

        DotString {
            head_ids: action_str.head_ids,
            tail_ids: action_str.tail_ids,
            body: body_str,
        }
    }
}

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

/**
 * Get second arg in action output
 */
#[derive(Debug)]
pub struct TupleSecond<T: Action, U> {
    action: T,
    _phantom_u: PhantomData<U>,
}

impl<T: Action, U> Action for TupleSecond<T, U> {}

/**
 * Implementation for the ActionWhile struct.  
 */
impl<T: Action, U> TupleSecond<T, U> {
    pub const fn new(action: T) -> Self {
        Self {
            action,
            _phantom_u: PhantomData,
        }
    }
}

#[async_trait]
impl<U: Send + Sync, V: Send + Sync, T: ActionExec<(U, V)>> ActionExec<V> for TupleSecond<T, U> {
    async fn execute(&mut self) -> V {
        self.action.execute().await.1
    }
}
