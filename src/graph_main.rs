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
    // (name, action) pairs to draw
    let actions = vec![("descend_and_go_forward", descend_and_go_forward(&context))];

    stream::iter(actions)
        .for_each(|(name, act)| async move {
            let (res1, res2) = join!(
                write(name.to_string() + ".svg", draw_svg(&act).unwrap()),
                write(name.to_string() + ".dot", dot_file(&act))
            );
            res1.unwrap();
            res2.unwrap();
        })
        .await;
}
