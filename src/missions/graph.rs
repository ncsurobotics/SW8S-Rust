use std::any::type_name;
use uuid::Uuid;

use super::action::Action;

pub fn stripped_type<T: ?Sized>() -> &'static str {
    let mut raw = type_name::<T>();
    // Remove generic signatures
    if let Some(idx) = raw.find('<') {
        raw = &raw[0..idx];
    }
    // Remove path
    if let Some(idx) = raw.rfind(':') {
        raw = &raw[idx + 1..];
    }
    raw
}

/// Contains information for dot (graphviz) graphing
#[derive(Debug)]
pub struct DotString {
    pub head_ids: Vec<Uuid>,
    pub tail_ids: Vec<Uuid>,
    pub body: String,
}

impl DotString {
    pub fn prepend(&mut self, other: &Self) {
        self.body = other.body.clone() + "\n" + &self.body;
    }

    pub fn append(&mut self, other: &Self) {
        self.body.push('\n');
        self.body.push_str(&other.body);
    }
}

pub fn dot_file<T: Action>(act: &T) -> String {
    let header = "digraph G {\nsplines = false;\n".to_string();
    header + &act.dot_string().body + "}"
}

#[cfg(test)]
mod tests {
    use crate::missions::{
        action_context::EmptyActionContext,
        example::{always_wait, initial_descent, race_conditional, sequence_conditional},
    };

    use super::*;

    #[test]
    fn dot_conditional() {
        let action = always_wait(&EmptyActionContext {});
        let file = dot_file(&action);
        println!("{file}");
    }

    #[test]
    fn dot_sequence_conditional() {
        let action = sequence_conditional(&EmptyActionContext {});
        let file = dot_file(&action);
        println!("{file}");
    }

    #[test]
    fn dot_race_conditional() {
        let action = race_conditional(&EmptyActionContext {});
        let file = dot_file(&action);
        println!("{file}");
    }

    #[test]
    fn dot_file_basic() {
        let action = initial_descent(&EmptyActionContext {});
        let file = dot_file(&action);
        println!("{file}");
    }
}
