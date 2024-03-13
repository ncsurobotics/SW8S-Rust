use anyhow::Result;
use async_trait::async_trait;
use core::fmt::Debug;
use std::{sync::Arc, thread};
use tokio::{join, runtime::Handle, sync::Mutex};
use uuid::Uuid;

use super::graph::{stripped_type, DotString};

/**
 * A trait for an action that can be executed.
 *
 * Functions returning Actions/ActionExec should be written with `-> impl ActionExec + '_` and
 * using `Con` for the context generic. This allows the build script to strip out the context and
 * create a mirrored version returning `-> impl Action +'_` for graphing.
 */
pub trait Action {
    /// Represent this node in dot (graphviz) notation
    fn dot_string(&self, _parent: &str) -> DotString {
        let id = Uuid::new_v4();
        DotString {
            head_ids: vec![id],
            tail_ids: vec![id],
            body: format!(
                "\"{}\" [label = \"{}\", margin = 0];\n",
                id,
                stripped_type::<Self>()
            ),
        }
    }
}

/**
 * A trait for an action that can be executed.
 */
#[async_trait]
pub trait ActionExec: Action + Send + Sync {
    type Output: Send + Sync;
    async fn execute(&mut self) -> Self::Output;
}

/**
 * A trait that can be executed and modified at runtime.
 */
pub trait ActionMod<Input: Send + Sync>: Action {
    fn modify(&mut self, input: &Input);
}

/**
 * Simplifies combining tuple actions
 * Takes input as `wrapper`, `child`, `child`,...
 *
 * `wrapper`: Tuple action to combine with (e.g. RaceAction)
 * `child`: Each action to be combined
 */
#[macro_export]
macro_rules! act_nest {
    ($wrapper:expr, $action_l:expr, $action_r:expr $(,)?) => {
        $wrapper($action_l, $action_r)
    };
    ($wrapper:expr, $action_l:expr, $( $action_r:expr $(,)? ),+) => {
       $wrapper($action_l, act_nest!($wrapper, $(
                   $action_r
           ),+))
    };
}

/**
 * Produces a new function that puts `wrapper` around `wrapee`.
 *
 * Needed for act_nest! macros that want to combine `MetaAction::new` calls,
 * since the recursive construction doesn't use identical values at each call site.
 *
 * e.g. `wrap_action(ActionConcurrent::new, FirstValid::new)` in order to run the
 * nested actions concurrently, outputting the first valid return
 */
pub fn wrap_action<T, U, V, W, X: Fn(T, U) -> V, Y: Fn(V) -> W>(
    wrapee: X,
    wrapper: Y,
) -> impl Fn(T, U) -> W {
    move |val1: T, val2: U| wrapper(wrapee(val1, val2))
}

/**
 * An action that runs one of two actions depending on if its conditional reference is true or false.  
 */
#[derive(Debug, Clone)]
pub struct ActionConditional<V: Action, W: Action, X: Action> {
    condition: V,
    true_branch: W,
    false_branch: X,
}

impl<V: Action, W: Action, X: Action> Action for ActionConditional<V, W, X> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let true_str = self.true_branch.dot_string(stripped_type::<Self>());
        let false_str = self.false_branch.dot_string(stripped_type::<Self>());
        let condition_str = self.condition.dot_string(stripped_type::<Self>());

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
impl<V: Action, W: Action, X: Action> ActionConditional<V, W, X> {
    pub const fn new(condition: V, true_branch: W, false_branch: X) -> Self {
        Self {
            condition,
            true_branch,
            false_branch,
        }
    }
}

/**
 * Implement the conditional logic for the ActionConditional action.
 */
#[async_trait]
impl<
        U: Send + Sync,
        V: ActionExec<Output = bool>,
        W: ActionExec<Output = U>,
        X: ActionExec<Output = U>,
    > ActionExec for ActionConditional<V, W, X>
{
    type Output = U;
    async fn execute(&mut self) -> Self::Output {
        if self.condition.execute().await {
            self.true_branch.execute().await
        } else {
            self.false_branch.execute().await
        }
    }
}

#[derive(Debug, Clone)]
/**
 * Action that runs two actions at the same time and exits both when one exits
 */
pub struct RaceAction<T: Action, U: Action> {
    first: T,
    second: U,
}

impl<T: Action, U: Action> Action for RaceAction<T, U> {
    fn dot_string(&self, parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());

        if parent == stripped_type::<Self>() {
            DotString {
                head_ids: [first_str.head_ids, second_str.head_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                tail_ids: [first_str.tail_ids, second_str.tail_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                body: first_str.body + &second_str.body,
            }
        } else {
            let race_id = Uuid::new_v4();
            let resolve_id = Uuid::new_v4();

            let mut body_str = format!(
            "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = red;\n\"{}\" [label = \"Race\", shape = box, fontcolor = red, style = dashed];\nstyle = dashed;\ncolor = red;\n\"{}\" [label = \"Resolve\", shape = box, fontcolor = red, style = dashed];\n",
            Uuid::new_v4(),
            race_id,
            resolve_id
        ) + &first_str.body
            + &second_str.body;

            [&first_str.head_ids, &second_str.head_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", race_id, id)));
            [&first_str.tail_ids, &second_str.tail_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{}\" -> \"{}\";\n", id, resolve_id)));
            body_str.push_str("}\n");

            DotString {
                head_ids: vec![race_id],
                tail_ids: vec![resolve_id],
                body: body_str,
            }
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
impl<V: Sync + Send, T: ActionExec<Output = V>, U: ActionExec<Output = V>> ActionExec
    for RaceAction<T, U>
{
    type Output = V;
    async fn execute(&mut self) -> Self::Output {
        tokio::select! {
            res = self.first.execute() => res,
            res = self.second.execute() => res
        }
    }
}

/**
 * Run two actions at once, and only exit when all actions have exited.
 */
#[derive(Debug, Clone)]
pub struct DualAction<T: Action, U: Action> {
    first: T,
    second: U,
}

impl<T: Action, U: Action> Action for DualAction<T, U> {
    fn dot_string(&self, parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());
        let (dual_head, dual_tail) = (Uuid::new_v4(), Uuid::new_v4());

        if parent == stripped_type::<Self>() {
            DotString {
                head_ids: [first_str.head_ids, second_str.head_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                tail_ids: [first_str.tail_ids, second_str.tail_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                body: first_str.body + &second_str.body,
            }
        } else {
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
impl<V: Send + Sync, T: ActionExec<Output = V>, U: ActionExec<Output = V>> ActionExec
    for DualAction<T, U>
{
    type Output = (V, V);
    async fn execute(&mut self) -> Self::Output {
        tokio::join!(self.first.execute(), self.second.execute())
    }
}

impl<Input: Send + Sync, V: ActionMod<Input> + Sync + Send, W: ActionMod<Input> + Sync + Send>
    ActionMod<Input> for DualAction<V, W>
{
    fn modify(&mut self, input: &Input) {
        self.first.modify(input);
        self.second.modify(input);
    }
}

#[derive(Debug, Clone)]
pub struct ActionChain<V: Action, W: Action> {
    first: V,
    second: W,
}

impl<V: Action, W: Action> Action for ActionChain<V, W> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());

        let mut body_str = first_str.body + &second_str.body;
        for tail in &first_str.tail_ids {
            for head in &second_str.head_ids {
                body_str.push_str(&format!(
                    "\"{}\" -> \"{}\" [color = purple, fontcolor = purple, label = \"Pass Data\"];\n",
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

impl<V: Action, W: Action> ActionChain<V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl<
        T: Send + Sync,
        U: Send + Sync,
        V: ActionExec<Output = T>,
        W: ActionMod<T> + ActionExec<Output = U>,
    > ActionExec for ActionChain<V, W>
{
    type Output = U;
    async fn execute(&mut self) -> Self::Output {
        self.second.modify(&self.first.execute().await);
        self.second.execute().await
    }
}

#[derive(Debug, Clone)]
pub struct ActionSequence<V, W> {
    first: V,
    second: W,
}

impl<V: Action, W: Action> Action for ActionSequence<V, W> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());

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

impl<V, W> ActionSequence<V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl<X: Send + Sync, V: ActionExec, W: ActionExec<Output = X>> ActionExec for ActionSequence<V, W> {
    type Output = X;
    async fn execute(&mut self) -> Self::Output {
        self.first.execute().await;
        self.second.execute().await
    }
}

#[derive(Debug, Clone)]
pub struct ActionParallel<V: Action, W: Action> {
    first: Arc<Mutex<V>>,
    second: Arc<Mutex<W>>,
}

impl<V: Action, W: Action> Action for ActionParallel<V, W> {
    fn dot_string(&self, parent: &str) -> DotString {
        let mut self_type = stripped_type::<Self>().to_string();
        if parent == "FirstValid" {
            self_type = self_type + "_" + parent;
        }

        let first_str = self.first.blocking_lock().dot_string(&self_type);
        let second_str = self.second.blocking_lock().dot_string(&self_type);
        let (par_head, par_tail) = (Uuid::new_v4(), Uuid::new_v4());

        if parent.contains(stripped_type::<Self>()) {
            DotString {
                head_ids: [first_str.head_ids, second_str.head_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                tail_ids: [first_str.tail_ids, second_str.tail_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                body: first_str.body + &second_str.body,
            }
        } else {
            let mut name = "Parallel";
            let mut color = "blue";
            if parent == "FirstValid" {
                name = "First Valid (Parallel)";
                color = "darkgreen";
            }

            let mut body_str = format!(
                "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = {};\n\"{}\" [label = \"{}\", shape = box, fontcolor = {}, style = dashed];\n",
                Uuid::new_v4(),
                color,
                par_head,
                name,
                color,
            ) + &format!("{}\" [label = \"Collect\", shape = box, fontcolor = {}, style = dashed];\n", par_tail, color) +
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
}

impl<V: Action, W: Action> ActionParallel<V, W> {
    pub fn new(first: V, second: W) -> Self {
        Self {
            first: Arc::from(Mutex::from(first)),
            second: Arc::from(Mutex::from(second)),
        }
    }
}

#[async_trait]
impl<
        Y: 'static + Send + Sync,
        X: 'static + Send + Sync,
        V: 'static + ActionExec<Output = Y>,
        W: 'static + ActionExec<Output = X>,
    > ActionExec for ActionParallel<V, W>
{
    type Output = (Y, X);
    async fn execute(&mut self) -> Self::Output {
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

#[derive(Debug, Clone)]
pub struct ActionConcurrent<V: Action, W: Action> {
    first: V,
    second: W,
}

impl<V: Action, W: Action> Action for ActionConcurrent<V, W> {
    fn dot_string(&self, parent: &str) -> DotString {
        let mut self_type = stripped_type::<Self>().to_string();
        if parent == "FirstValid" {
            self_type = self_type + "_" + parent;
        }

        let first_str = self.first.dot_string(&self_type);
        let second_str = self.second.dot_string(&self_type);
        let (concurrent_head, concurrent_tail) = (Uuid::new_v4(), Uuid::new_v4());

        if parent.contains(stripped_type::<Self>()) {
            DotString {
                head_ids: [first_str.head_ids, second_str.head_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                tail_ids: [first_str.tail_ids, second_str.tail_ids]
                    .into_iter()
                    .flatten()
                    .collect(),
                body: first_str.body + &second_str.body,
            }
        } else {
            let mut name = "Concurrent";
            let mut color = "blue";
            if parent == "FirstValid" {
                name = "First Valid (Concurrent)";
                color = "darkgreen";
            }

            let mut body_str = format!(
                "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = {};\n\"{}\" [label = \"{}\", shape = box, fontcolor = {}, style = dashed];\n",
                Uuid::new_v4(),
                color,
                concurrent_head,
                name,
                color,
            );

            body_str.push_str(&(first_str.body + &second_str.body));
            vec![first_str.head_ids, second_str.head_ids]
                .into_iter()
                .flatten()
                .for_each(|id| {
                    body_str.push_str(&format!("\"{}\" -> \"{}\";\n", concurrent_head, id))
                });

            let tail_ids = if parent != "TupleSecond" {
                body_str.push_str(&(format!("\"{}\" [label = \"Converge\", shape = box, fontcolor = {}, style = dashed];\n", concurrent_tail, color)));
                vec![first_str.tail_ids, second_str.tail_ids.clone()]
                    .into_iter()
                    .flatten()
                    .for_each(|id| {
                        body_str.push_str(&format!("\"{}\" -> \"{}\";\n", id, concurrent_tail))
                    });
                vec![concurrent_tail]
            } else {
                second_str.tail_ids
            };

            body_str.push_str("}\n");

            DotString {
                head_ids: vec![concurrent_head],
                tail_ids,
                body: body_str,
            }
        }
    }
}

impl<V: Action, W: Action> ActionConcurrent<V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl<X: Send + Sync, Y: Send + Sync, V: ActionExec<Output = Y>, W: ActionExec<Output = X>>
    ActionExec for ActionConcurrent<V, W>
{
    type Output = (Y, X);
    async fn execute(&mut self) -> Self::Output {
        join!(self.first.execute(), self.second.execute())
    }
}

impl<Input: Send + Sync, V: ActionMod<Input> + Sync + Send, W: ActionMod<Input> + Sync + Send>
    ActionMod<Input> for ActionConcurrent<V, W>
{
    fn modify(&mut self, input: &Input) {
        self.first.modify(input);
        self.second.modify(input);
    }
}

/**
 * An action that tries `count` times for a success
 */
#[derive(Debug, Clone)]
pub struct ActionUntil<T: Action> {
    action: T,
    limit: u32,
}

impl<T: Action> Action for ActionUntil<T> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let action_str = self.action.dot_string(stripped_type::<Self>());

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
impl<U: Send + Sync, T: ActionExec<Output = Result<U>>> ActionExec for ActionUntil<T> {
    type Output = Result<U>;
    async fn execute(&mut self) -> Self::Output {
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
#[derive(Debug, Clone)]
pub struct ActionWhile<T: Action> {
    action: T,
}

impl<T: Action> Action for ActionWhile<T> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let action_str = self.action.dot_string(stripped_type::<Self>());

        let mut body_str = action_str.body;
        for head in &action_str.head_ids {
            for tail in &action_str.tail_ids {
                body_str.push_str(&format!("\"{}\" [shape = diamond];\n", tail));
                body_str.push_str(&format!(
                    "\"{}\" -> \"{}\" [label = \"True\"];\n",
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
impl<U: Send + Sync, T: ActionExec<Output = Result<U>>> ActionExec for ActionWhile<T> {
    type Output = Result<U>;
    async fn execute(&mut self) -> Self::Output {
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
#[derive(Debug, Clone)]
pub struct TupleSecond<T: Action> {
    action: T,
}

impl<T: Action> Action for TupleSecond<T> {
    fn dot_string(&self, _parent: &str) -> DotString {
        self.action.dot_string(stripped_type::<Self>())
    }
}

/**
 * Implementation for the ActionWhile struct.
 */
impl<T: Action> TupleSecond<T> {
    pub const fn new(action: T) -> Self {
        Self { action }
    }
}

#[async_trait]
impl<U: Send + Sync, V: Send + Sync, T: ActionExec<Output = (U, V)>> ActionExec for TupleSecond<T> {
    type Output = V;
    async fn execute(&mut self) -> Self::Output {
        self.action.execute().await.1
    }
}

impl<Input: Send + Sync, V: ActionMod<Input> + Sync + Send> ActionMod<Input> for TupleSecond<V> {
    fn modify(&mut self, input: &Input) {
        self.action.modify(input);
    }
}

/**
 * Return first valid response from block of actions.
 */
#[derive(Debug, Clone)]
pub struct FirstValid<T: Action> {
    action: T,
}

impl<T: Action> Action for FirstValid<T> {
    fn dot_string(&self, parent: &str) -> DotString {
        let mut self_type = stripped_type::<Self>();
        if parent.contains(self_type) {
            self_type = parent;
        }
        self.action.dot_string(self_type)
    }
}

/**
 * Implementation for the FirstValid struct.  
 */
impl<T: Action> FirstValid<T> {
    pub const fn new(action: T) -> Self {
        Self { action }
    }
}

impl<Input: Send + Sync, T: ActionMod<Input> + Sync + Send> ActionMod<Input> for FirstValid<T> {
    fn modify(&mut self, input: &Input) {
        self.action.modify(input);
    }
}

#[async_trait]
impl<U: Send + Sync, T: ActionExec<Output = (Result<U>, Result<U>)>> ActionExec for FirstValid<T> {
    type Output = Result<U>;
    async fn execute(&mut self) -> Self::Output {
        let (first, second) = self.action.execute().await;
        if first.is_ok() {
            first
        } else {
            second
        }
    }
}
