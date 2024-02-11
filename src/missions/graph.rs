use std::any::type_name;
use uuid::Uuid;

#[cfg(feature = "graphing")]
use graphviz_rust::{cmd::Format, exec, parse, printer::PrinterContext};

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
#[derive(Debug, Clone)]
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

pub fn dot_file<T: ?Sized + Action>(act: &T) -> String {
    let header = "digraph G {\nsplines = true;\nnodesep = 1.0;\nbgcolor = \"none\";\n".to_string();
    header + &act.dot_string("").body + "}"
}

#[cfg(feature = "graphing")]
pub fn draw_svg<T: ?Sized + Action>(act: &T) -> std::io::Result<Vec<u8>> {
    exec(
        parse(&dot_file(act)).unwrap(),
        &mut PrinterContext::default(),
        vec![Format::Svg.into()],
    )
}
