use std::fs::create_dir_all;

use futures::{stream, StreamExt};
use sw8s_rust_lib::missions::action_context::EmptyActionContext;
use sw8s_rust_lib::missions::graph::{dot_file, draw_svg};
use tokio::{fs::write, join};

#[allow(warnings)]
mod generated_actions {
    include!(concat!(env!("OUT_DIR"), "/graph_missions/basic.rs"));
}
use generated_actions::descend_and_go_forward;

#[tokio::main]
async fn main() {
    let context = EmptyActionContext;
    create_dir_all("graphs/").unwrap();
    // (name, action) pairs to draw
    let actions = vec![("descend_and_go_forward", descend_and_go_forward(&context))];

    stream::iter(actions)
        .for_each(|(name, act)| async move {
            let (res1, res2) = join!(
                write(
                    "graphs/".to_string() + &name + ".svg",
                    draw_svg(&act).unwrap()
                ),
                write("graphs/".to_string() + &name + ".dot", dot_file(&act))
            );
            res1.unwrap();
            res2.unwrap();
        })
        .await;
}
