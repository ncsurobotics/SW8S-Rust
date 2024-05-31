use itertools::Itertools;
use std::any::type_name;
use uuid::Uuid;

#[cfg(feature = "graphing")]
use graphviz_rust::{cmd::Format, exec, parse, printer::PrinterContext};

use super::action::Action;

pub fn stripped_type<T: ?Sized>() -> &'static str {
    let raw = type_name::<T>();
    strip_sig(raw)
}

pub fn stripped_fn<T: ?Sized>() -> String {
    let raw = type_name::<T>();
    if raw.ends_with("{{closure}}") {
        "Anon Closure".to_string()
    } else {
        let (fn_sig, output) = raw.split("->").collect_tuple().unwrap_or((raw, "?"));
        let (fn_first, fn_mid, fn_end) = fn_sig.split(['(', ')']).collect_tuple().unwrap();
        fn_first.to_string() + "(" + strip_sig(fn_mid) + ")" + fn_end + "->" + strip_sig(output)
    }
}

fn strip_sig(mut raw: &str) -> &str {
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

/// Generate the .dot (graphviz) file to draw the action
pub fn dot_file<T: ?Sized + Action>(act: &T) -> String {
    let header = "digraph G {\nsplines = true;\nnodesep = 1.0;\nbgcolor = \"none\"\n".to_string();
    let full = header + &act.dot_string("").body + "}";
    indent_lines(reorder_dot_strings(&full.split('\n').collect_vec()).lines())
}

/// Reorders the generated dot strings to render better
///
/// For some reason the layout engine does better with labels declared after
/// lines, see <https://stackoverflow.com/a/9238952>.
/// This function reorders each graph and subgraph in the order of header, metadata, subgraphs,
/// edges, and then labels.
fn reorder_dot_strings(arr: &[&str]) -> String {
    let mut labels = vec![];
    let mut graphs = vec![];
    let mut edges = vec![];
    let mut meta = vec![];

    let header = arr.first().unwrap().to_string();
    let mut pos = 1;

    while pos < arr.len() {
        let val = arr[pos];
        pos += 1;
        if val.starts_with('}') {
            break;
        };
        if val.starts_with("subgraph") {
            let subgraph = reorder_dot_strings(&arr[(pos - 1)..]);
            pos += subgraph
                .split('\n')
                .filter(|line| !line.starts_with('}'))
                .count()
                - 1;
            graphs.push(subgraph);
            continue;
        };
        if let Some(stripped_val) = val.strip_prefix('"') {
            if let Some(end_quote) = stripped_val.find('"') {
                let end_pos = end_quote + 1;
                if &val[end_pos + 1..end_pos + 8] == " [label" {
                    labels.push(val);
                    continue;
                }
            }
        }
        if val.contains("->") {
            edges.push(val);
            continue;
        }
        // Only if one of the prior checks fails
        meta.push(val);
    }

    [
        vec![header],
        meta.into_iter().map(|x| x.to_string()).collect(),
        graphs,
        edges.into_iter().map(|x| x.to_string()).collect(),
        labels.into_iter().map(|x| x.to_string()).collect(),
        vec!["}".to_string()],
    ]
    .into_iter()
    .flatten()
    .fold(String::new(), |acc, line| (acc + &line + "\n"))
}

/// Indents blocks inside the .dot (graphviz) file
fn indent_lines<I, T>(iter: I) -> String
where
    I: IntoIterator<Item = T>,
    T: AsRef<str>,
{
    let mut num_tabs = 0;
    iter.into_iter()
        .map(|line| {
            let mut line = line.as_ref().to_string();
            if line.trim().starts_with('}') {
                num_tabs -= 1;
            }
            for _ in 0..num_tabs {
                line = "\t".to_string() + &line;
            }
            if line.ends_with('{') {
                num_tabs += 1;
            }
            line
        })
        .fold(String::new(), |acc, line| (acc + &line + "\n"))
}

#[cfg(feature = "graphing")]
pub fn draw_svg<T: ?Sized + Action>(act: &T) -> std::io::Result<Vec<u8>> {
    exec(
        parse(&dot_file(act)).unwrap(),
        &mut PrinterContext::default(),
        vec![Format::Svg.into()],
    )
}
