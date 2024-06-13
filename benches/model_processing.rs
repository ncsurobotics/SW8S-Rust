use criterion::{black_box, criterion_group, criterion_main, Criterion};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use sw8s_rust_lib::vision::{gate_poles::GatePoles, VisualDetector};

fn gate_pole_model(c: &mut Criterion) {
    const CUDA_ENABLED: &str = if cfg!(feature = "cuda") {
        if cfg!(feature = "cuda_f16") {
            "CUDA F16"
        } else {
            "CUDA"
        }
    } else {
        "CPU"
    };

    let image = imread(
        "tests/vision/resources/gate_images/straight_on_0.png",
        IMREAD_COLOR,
    )
    .unwrap();
    let mut model = GatePoles::default();

    c.bench_function(
        &("Gate Pole Model (".to_string() + CUDA_ENABLED + ")"),
        |b| {
            b.iter(|| {
                black_box(model.detect(&image).unwrap());
            })
        },
    );
}

criterion_group!(model_processing, gate_pole_model);
criterion_main!(model_processing);
