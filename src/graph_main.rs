use std::fs::create_dir_all;

use futures::{stream, StreamExt};
use paste::paste;
use sw8s_rust_lib::missions::action_context::EmptyActionContext;
use sw8s_rust_lib::missions::graph::{dot_file, draw_svg};
use tokio::{fs::write, join};

#[allow(warnings)]
mod generated_actions {
    pub mod basic {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/basic.rs"));
    }
    pub mod example {
        include!(concat!(env!("OUT_DIR"), "/graph_missions/example.rs"));
    }
}

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
    let actions = graph_actions!(basic, example);

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
