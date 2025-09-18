use anyhow::Result;

use core::fmt::Debug;
use std::{marker::PhantomData, sync::Arc, thread};
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

pub trait ActionIgnoredGeneric<T>: Action {}

impl<T, U: Action> ActionIgnoredGeneric<T> for U {}

/**
 * A trait for an action that can be executed.
 */
#[allow(async_fn_in_trait)]
pub trait ActionExec<T: Send + Sync>: Action + Send + Sync {
    async fn execute(&mut self) -> T;
}

/**
 * An action that can be modified at runtime.
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
            combined_str.push_str(&format!("\"{tail_id}\" [shape = diamond];\n"));
            for head_id in &true_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{tail_id}\" -> \"{head_id}\" [label = \"True\"];\n"
                ));
            }
            for head_id in &false_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{tail_id}\" -> \"{head_id}\" [label = \"False\"];\n",
                ));
            }
        }
        DotString {
            head_ids: condition_str.head_ids,
            tail_ids: vec![true_str.tail_ids, false_str.tail_ids]
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
impl<U: Send + Sync, V: ActionExec<bool>, W: ActionExec<U>, X: ActionExec<U>> ActionExec<U>
    for ActionConditional<V, W, X>
{
    async fn execute(&mut self) -> U {
        if self.condition.execute().await {
            self.true_branch.execute().await
        } else {
            self.false_branch.execute().await
        }
    }
}

/**
 * An action that runs one of two actions depending on if its conditional
 * reference is valid or not, passing processed output to the first child and
 * unprocessed output to the second child.
 */
#[derive(Debug, Clone)]
pub struct ActionDataConditional<V: Action, W: Action, X: Action, T, Y> {
    condition: V,
    true_branch: W,
    false_branch: X,
    _phantom: (PhantomData<T>, PhantomData<Y>),
}

impl<V: Action, W: Action, X: Action, T, Y> Action for ActionDataConditional<V, W, X, T, Y> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let true_str = self.true_branch.dot_string(stripped_type::<Self>());
        let false_str = self.false_branch.dot_string(stripped_type::<Self>());
        let condition_str = self.condition.dot_string(stripped_type::<Self>());

        let mut combined_str = true_str.body + &false_str.body + &condition_str.body;
        for tail_id in &condition_str.tail_ids {
            combined_str.push_str(&format!("\"{tail_id}\" [shape = diamond];\n"));
            for head_id in &true_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{tail_id}\" -> \"{head_id}\" [color = purple, fontcolor = purple, label = \"True (Pass Data)\"];\n"
                ));
            }
            for head_id in &false_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{tail_id}\" -> \"{head_id}\" [label = \"False\"];\n"
                ));
            }
        }
        for condition_head_id in &condition_str.head_ids {
            for head_id in &false_str.head_ids {
                combined_str.push_str(&format!(
                    "\"{condition_head_id}\" -> \"{head_id}\" [color = purple, fontcolor = purple, label = \"Pass Data\"];\n"
                ));
            }
        }
        DotString {
            head_ids: condition_str.head_ids,
            tail_ids: vec![true_str.tail_ids, false_str.tail_ids]
                .into_iter()
                .flatten()
                .collect(),
            body: combined_str,
        }
    }
}

/**
 * Implementation for the ActionDataConditional struct.
 */
impl<V: Action, W: Action, X: Action, T, Y> ActionDataConditional<V, W, X, T, Y> {
    pub const fn new(condition: V, true_branch: W, false_branch: X) -> Self {
        Self {
            condition,
            true_branch,
            false_branch,
            _phantom: (PhantomData, PhantomData),
        }
    }
}

/**
 * Implement the passing and conditional logic.
 */
impl<
        T: Send + Sync,
        Input: Send + Sync,
        U: Send + Sync,
        V: ActionExec<Option<T>> + ActionMod<Input>,
        W: ActionExec<U> + ActionMod<T>,
        X: ActionExec<U> + ActionMod<Input>,
    > ActionExec<U> for ActionDataConditional<V, W, X, T, Input>
{
    async fn execute(&mut self) -> U {
        if let Some(output) = self.condition.execute().await {
            self.true_branch.modify(&output);
            self.true_branch.execute().await
        } else {
            self.false_branch.execute().await
        }
    }
}

impl<
        V: ActionMod<Input> + Sync + Send,
        W: Action,
        X: ActionMod<Input> + Sync + Send,
        T,
        Input: Send + Sync,
    > ActionMod<Input> for ActionDataConditional<V, W, X, T, Input>
{
    fn modify(&mut self, input: &Input) {
        self.condition.modify(input);
        self.false_branch.modify(input);
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
                .for_each(|id| body_str.push_str(&format!("\"{race_id}\" -> \"{id}\";\n")));
            [&first_str.tail_ids, &second_str.tail_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{id}\" -> \"{resolve_id}\";\n")));
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
            "subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = blue;\n\"{dual_head}\" [label = \"Dual\", shape = box, fontcolor = blue, style = dashed];\n",
            Uuid::new_v4()) + &format!("{dual_tail}\" [label = \"Collect\", shape = box, fontcolor = blue, style = dashed];\n") +
            &first_str.body
            + &second_str.body;

            vec![first_str.head_ids, second_str.head_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{dual_head}\" -> \"{id}\";\n")));
            vec![first_str.tail_ids, second_str.tail_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{id}\" -> \"{dual_tail}\";\n")));
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
impl<V: Send + Sync, T: ActionExec<V>, U: ActionExec<V>> ActionExec<(V, V)> for DualAction<T, U> {
    async fn execute(&mut self) -> (V, V) {
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
pub struct ActionChain<T, V: Action, W: Action> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
}

impl<T, V: Action, W: Action> Action for ActionChain<T, V, W> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());

        let mut body_str = first_str.body + &second_str.body;
        for tail in &first_str.tail_ids {
            for head in &second_str.head_ids {
                body_str.push_str(&format!("\"{tail}\" -> \"{head}\" [color = purple, fontcolor = purple, label = \"Pass Data\"];\n"))
            }
        }

        DotString {
            head_ids: first_str.head_ids,
            tail_ids: second_str.tail_ids,
            body: body_str,
        }
    }
}

impl<T, V: Action, W: Action> ActionChain<T, V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
        }
    }
}

impl<T: Send + Sync, U: Send + Sync, V: ActionExec<T>, W: ActionMod<T> + ActionExec<U>>
    ActionExec<U> for ActionChain<T, V, W>
{
    async fn execute(&mut self) -> U {
        self.second.modify(&self.first.execute().await);
        self.second.execute().await
    }
}

impl<Input: Send + Sync, T, U: ActionMod<Input>, V: Action> ActionMod<Input>
    for ActionChain<T, U, V>
{
    fn modify(&mut self, input: &Input) {
        self.first.modify(input);
    }
}

#[derive(Debug, Clone)]
pub struct ActionSequence<T, V, W> {
    first: V,
    second: W,
    _phantom_t: PhantomData<T>,
}

impl<T, V: Action, W: Action> Action for ActionSequence<T, V, W> {
    fn dot_string(&self, _parent: &str) -> DotString {
        let first_str = self.first.dot_string(stripped_type::<Self>());
        let second_str = self.second.dot_string(stripped_type::<Self>());

        let mut label = "";
        let first_type = stripped_type::<V>();
        if ["ActionWhile", "ActionDataConditional", "ActionConditional"].contains(&first_type) {
            label = "[label = \"False\"]";
        }

        let mut body_str = first_str.body + &second_str.body;
        for tail in &first_str.tail_ids {
            for head in &second_str.head_ids {
                body_str.push_str(&format!("\"{tail}\" -> \"{head}\" {label};\n"))
            }
        }

        DotString {
            head_ids: if first_type != "Terminal" {
                first_str.head_ids
            } else {
                second_str.head_ids
            },
            tail_ids: second_str.tail_ids,
            body: body_str,
        }
    }
}

impl<T, V, W> ActionSequence<T, V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
        }
    }
}

impl<T: Send + Sync, X: Send + Sync, V: ActionExec<T>, W: ActionExec<X>> ActionExec<X>
    for ActionSequence<T, V, W>
{
    async fn execute(&mut self) -> X {
        self.first.execute().await;
        self.second.execute().await
    }
}

impl<T: Send + Sync, X: Send + Sync, V: ActionMod<X>, W: Action> ActionMod<X>
    for ActionSequence<T, V, W>
{
    fn modify(&mut self, input: &X) {
        self.first.modify(input)
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
        if parent.contains("FirstValid") && (parent == "FirstValid" || parent.contains("Parallel"))
        {
            self_type += "_FirstValid";
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
                name = "FirstValid (Parallel)";
                color = "darkgreen";
            }

            let mut body_str = format!("subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = {color};\n\"{par_head}\" [label = \"{name}\", shape = box, fontcolor = {color}, style = dashed];\n", Uuid::new_v4())
                + &format!("{par_tail}\" [label = \"Collect\", shape = box, fontcolor = {color}, style = dashed];\n")
                + &first_str.body
                + &second_str.body;

            vec![first_str.head_ids, second_str.head_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{par_head}\" -> \"{id}\";\n")));
            vec![first_str.tail_ids, second_str.tail_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{id}\" -> \"{par_tail}\";\n")));
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

impl<
        Y: 'static + Send + Sync,
        X: 'static + Send + Sync,
        V: 'static + ActionExec<Y>,
        W: 'static + ActionExec<X>,
    > ActionExec<(Y, X)> for ActionParallel<V, W>
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

#[derive(Debug, Clone)]
pub struct ActionConcurrent<V: Action, W: Action> {
    first: V,
    second: W,
}

impl<V: Action, W: Action> Action for ActionConcurrent<V, W> {
    fn dot_string(&self, parent: &str) -> DotString {
        let mut self_type = stripped_type::<Self>().to_string();
        if parent.contains("FirstValid")
            && (parent == "FirstValid" || parent.contains("Concurrent"))
        {
            self_type += "_FirstValid";
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
                name = "FirstValid (Concurrent)";
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
                .for_each(|id| body_str.push_str(&format!("\"{concurrent_head}\" -> \"{id}\";\n")));

            let tail_ids = if parent != "TupleSecond" {
                body_str.push_str(&(format!("\"{concurrent_tail}\" [label = \"Converge\", shape = box, fontcolor = {color}, style = dashed];\n")));
                vec![first_str.tail_ids, second_str.tail_ids.clone()]
                    .into_iter()
                    .flatten()
                    .for_each(|id| {
                        body_str.push_str(&format!("\"{id}\" -> \"{concurrent_tail}\";\n"))
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

impl<X: Send + Sync, Y: Send + Sync, V: ActionExec<Y>, W: ActionExec<X>> ActionExec<(Y, X)>
    for ActionConcurrent<V, W>
{
    async fn execute(&mut self) -> (Y, X) {
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

#[derive(Debug, Clone)]
pub struct ActionConcurrentSplit<V: Action, W: Action> {
    first: V,
    second: W,
}

impl<V: Action, W: Action> Action for ActionConcurrentSplit<V, W> {
    fn dot_string(&self, parent: &str) -> DotString {
        let mut self_type = stripped_type::<Self>().to_string();
        if parent.contains("FirstValid")
            && (parent == "FirstValid" || parent.contains("Concurrent"))
        {
            self_type += "_FirstValid";
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
                name = "FirstValid (Concurrent)";
                color = "darkgreen";
            }

            let mut body_str = format!("subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = {color};\n\"{concurrent_head}\" [label = \"{name}\", shape = box, fontcolor = {color}, style = dashed];\n", Uuid::new_v4());

            body_str.push_str(&(first_str.body + &second_str.body));
            vec![first_str.head_ids, second_str.head_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{concurrent_head}\" -> \"{id}\";\n")));

            let tail_ids = if parent != "TupleSecond" {
                body_str.push_str(&(format!("\"{concurrent_tail}\" [label = \"Converge\", shape = box, fontcolor = {color}, style = dashed];\n")));
                vec![first_str.tail_ids, second_str.tail_ids.clone()]
                    .into_iter()
                    .flatten()
                    .for_each(|id| {
                        body_str.push_str(&format!("\"{id}\" -> \"{concurrent_tail}\";\n"))
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

impl<V: Action, W: Action> ActionConcurrentSplit<V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self { first, second }
    }
}

impl<X: Send + Sync, Y: Send + Sync, V: ActionExec<Y>, W: ActionExec<X>> ActionExec<(Y, X)>
    for ActionConcurrentSplit<V, W>
{
    async fn execute(&mut self) -> (Y, X) {
        join!(self.first.execute(), self.second.execute())
    }
}

impl<
        InputLhs: Send + Sync,
        InputRhs: Send + Sync,
        V: ActionMod<InputLhs> + Sync + Send,
        W: ActionMod<InputRhs> + Sync + Send,
    > ActionMod<(InputLhs, InputRhs)> for ActionConcurrentSplit<V, W>
{
    fn modify(&mut self, input: &(InputLhs, InputRhs)) {
        self.first.modify(&input.0);
        self.second.modify(&input.1);
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
            body_str.push_str(&format!("\"{head}\" [shape = diamond];\n"));
            for tail in &action_str.tail_ids {
                body_str.push_str(&format!(
                    "\"{tail}\":sw -> \"{head}\":nw [label = \"Fail Within Count\"];\n"
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
                body_str.push_str(&format!("\"{tail}\" [shape = diamond];\n"));
                body_str.push_str(&format!("\"{tail}\" -> \"{head}\" [label = \"True\"];\n"))
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

impl<U: Send + Sync + Default, T: ActionExec<Result<U>>> ActionExec<U> for ActionWhile<T> {
    async fn execute(&mut self) -> U {
        let mut result = U::default();
        loop {
            if let Ok(new_result) = self.action.execute().await {
                result = new_result;
            } else {
                return result;
            }
        }
    }
}

/**
 * Get second arg in action output
 */
#[derive(Debug, Clone)]
pub struct TupleSecond<T: Action, U> {
    action: T,
    _phantom_u: PhantomData<U>,
}

impl<T: Action, U> Action for TupleSecond<T, U> {
    fn dot_string(&self, _parent: &str) -> DotString {
        self.action.dot_string(stripped_type::<Self>())
    }
}

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

impl<U: Send + Sync, V: Send + Sync, T: ActionExec<(U, V)>> ActionExec<V> for TupleSecond<T, U> {
    async fn execute(&mut self) -> V {
        self.action.execute().await.1
    }
}

impl<Input: Send + Sync, V: ActionMod<Input> + Sync + Send, U> ActionMod<Input>
    for TupleSecond<V, U>
{
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

impl<U: Send + Sync, T: ActionExec<(Result<U>, Result<U>)>> ActionExec<Result<U>>
    for FirstValid<T>
{
    async fn execute(&mut self) -> Result<U> {
        let (first, second) = self.action.execute().await;
        if first.is_ok() {
            first
        } else {
            second
        }
    }
}

impl<U: Send + Sync, T: ActionExec<(Option<U>, Option<U>)>> ActionExec<Option<U>>
    for FirstValid<T>
{
    async fn execute(&mut self) -> Option<U> {
        let (first, second) = self.action.execute().await;
        if first.is_some() {
            first
        } else {
            second
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionSelect<V: Action, W: Action> {
    first: V,
    second: W,
}

impl<V: Action, W: Action> Action for ActionSelect<V, W> {
    fn dot_string(&self, parent: &str) -> DotString {
        let mut self_type = stripped_type::<Self>().to_string();
        if parent.contains("FirstValid") && (parent == "FirstValid" || parent.contains("Select")) {
            self_type += "_FirstValid";
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
            let mut name = "Select";
            let mut color = "blue";
            if parent == "FirstValid" {
                name = "FirstValid (Select)";
                color = "darkgreen";
            }

            let mut body_str = format!("subgraph \"cluster_{}\" {{\nstyle = dashed;\ncolor = {color};\n\"{concurrent_head}\" [label = \"{name}\", shape = box, fontcolor = {color}, style = dashed];\n", Uuid::new_v4());

            body_str.push_str(&(first_str.body + &second_str.body));
            vec![first_str.head_ids, second_str.head_ids]
                .into_iter()
                .flatten()
                .for_each(|id| body_str.push_str(&format!("\"{concurrent_head}\" -> \"{id}\";\n")));

            let tail_ids = if parent != "TupleSecond" {
                body_str.push_str(&(format!("\"{concurrent_tail}\" [label = \"Converge\", shape = box, fontcolor = {color}, style = dashed];\n")));
                vec![first_str.tail_ids, second_str.tail_ids.clone()]
                    .into_iter()
                    .flatten()
                    .for_each(|id| {
                        body_str.push_str(&format!("\"{id}\" -> \"{concurrent_tail}\";\n"))
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

impl<V: Action, W: Action> ActionSelect<V, W> {
    pub const fn new(first: V, second: W) -> Self {
        Self { first, second }
    }
}

impl<X: Send + Sync, V: ActionExec<X>, W: ActionExec<X>> ActionExec<X> for ActionSelect<V, W> {
    async fn execute(&mut self) -> X {
        tokio::select!(x = self.first.execute() => x, x = self.second.execute() => x)
    }
}

impl<Input: Send + Sync, V: ActionMod<Input> + Sync + Send, W: ActionMod<Input> + Sync + Send>
    ActionMod<Input> for ActionSelect<V, W>
{
    fn modify(&mut self, input: &Input) {
        self.first.modify(input);
        self.second.modify(input);
    }
}
