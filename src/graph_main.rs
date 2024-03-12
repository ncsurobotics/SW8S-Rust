use std::fs::create_dir_all;

use futures::{stream, StreamExt};
use paste::paste;
use tokio::{fs::write, join};

#[allow(warnings)]
pub mod generated_actions {
    pub mod basic {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/basic.rs"));
    }
    pub mod example {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/example.rs"));
    }
    pub mod buoy_hit {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/buoy_hit.rs"));
    }
    pub mod buoy_circle {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/buoy_circle.rs"));
    }
    pub mod path_align {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/path_align.rs"));
    }
    pub mod gate {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/gate.rs"));
    }

    // TODO: find some way to automate the extras
    pub mod action_context {
        include!(concat!(
            env!("OUT_DIR"),
            "/graph_missions/action_context.rs"
        ));
    }
    pub mod vision {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/vision.rs"));
    }
    pub mod movement {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/movement.rs"));
    }
    pub mod meb {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/meb.rs"));
    }
    pub mod comms {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/comms.rs"));
    }
    pub mod graph {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/graph.rs"));
    }
    pub mod action {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/action.rs"));
    }
}

use generated_actions::action_context::EmptyActionContext;
use generated_actions::graph::{dot_file, draw_svg};

const CONTEXT: EmptyActionContext = EmptyActionContext;

macro_rules! graph_actions {
    ($($i:path),*) => {
        vec![
            $(
            paste! {
                (stringify!($i), generated_actions::[<$i>]::graph_actions(&CONTEXT))
        }
            ),*
        ]
    }
}

#[tokio::main]
async fn main() {
    create_dir_all("graphs/").unwrap();
    // (name, action) pairs to draw
    let actions = graph_actions!(basic, example, buoy_hit, buoy_circle, path_align, gate);

    stream::iter(actions)
        .for_each(|(dir_name, action_set)| async move {
            let dir = ("graphs/".to_string() + dir_name + "/").clone();
            tokio::fs::create_dir_all(dir.clone()).await.unwrap();
            stream::iter(action_set)
                .map(|(name, act)| (dir.clone(), name, act))
                .for_each(|(dir, name, act)| async move {
                    let (res1, res2) = join!(
                        write(dir.clone() + &name + ".svg", draw_svg(&*act).unwrap()),
                        write(dir.clone() + &name + ".dot", dot_file(&*act))
                    );
                    res1.unwrap();
                    res2.unwrap();
                })
                .await;
        })
        .await;
}
