use criterion::{black_box, criterion_group, criterion_main, Criterion};
use opencv::{
    core::Vector,
    imgcodecs::{imread, imwrite, IMREAD_COLOR},
};
use sw8s_rust_lib::vision::{gate_poles::GatePoles, Draw, VisualDetector};

fn gate_pole_model(c: &mut Criterion) {
    const CUDA_ENABLED: &str = if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "CPU"
    };

    let image = imread(
        "tests/vision/resources/gate_images/vlcsnap-2023-08-04-17h21m51s095.png",
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
