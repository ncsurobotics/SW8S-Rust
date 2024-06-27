use std::num::NonZeroUsize;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use sw8s_rust_lib::vision::{
    buoy_model::BuoyModel, gate_poles::GatePoles, nn_cv2::ModelPipelined, MatWrapper,
    VisualDetector,
};

const CUDA_ENABLED: &str = if cfg!(feature = "cuda") {
    if cfg!(feature = "cuda_f16") {
        "CUDA F16"
    } else {
        "CUDA"
    }
} else {
    "CPU"
};

fn gate_pole_model(c: &mut Criterion) {
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

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let pipeline_model = runtime.block_on(GatePoles::default().into_pipelined(
        NonZeroUsize::try_from(2).unwrap(),
        NonZeroUsize::try_from(2).unwrap(),
    ));

    c.bench_function(
        &("Gate Pole Model Pipelined (".to_string() + CUDA_ENABLED + ")"),
        |b| {
            b.to_async(&runtime).iter(|| async {
                pipeline_model.update_mat(MatWrapper(image.clone()));
                black_box(pipeline_model.get_single().await);
            })
        },
    );
}

fn buoy_model(c: &mut Criterion) {
    let image = imread(
        "tests/vision/resources/buoy_images/straight_on_0.png",
        IMREAD_COLOR,
    )
    .unwrap();
    let mut model = BuoyModel::default();

    c.bench_function(&("Buoy Model (".to_string() + CUDA_ENABLED + ")"), |b| {
        b.iter(|| {
            black_box(model.detect(&image).unwrap());
        })
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let pipeline_model = runtime.block_on(BuoyModel::default().into_pipelined(
        NonZeroUsize::try_from(2).unwrap(),
        NonZeroUsize::try_from(2).unwrap(),
    ));

    c.bench_function(
        &("Buoy Model Pipelined (".to_string() + CUDA_ENABLED + ")"),
        |b| {
            b.to_async(&runtime).iter(|| async {
                pipeline_model.update_mat(MatWrapper(image.clone()));
                black_box(pipeline_model.get_single().await);
            })
        },
    );
}

criterion_group!(model_processing, gate_pole_model, buoy_model);
criterion_main!(model_processing);
