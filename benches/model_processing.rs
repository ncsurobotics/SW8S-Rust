use std::{num::NonZeroUsize, sync::Arc, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use sw8s_rust_lib::vision::{
    buoy_model::BuoyModel,
    gate_poles::GatePoles,
    nn_cv2::{ModelPipelined, OnnxModel, VisionModel},
    MatWrapper, VisualDetector,
};
use tokio::time::sleep;

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

fn pipelined(c: &mut Criterion) {
    const MAX_MODEL_THREADS: usize = 6;
    const MAX_POST_PROCESSING_THREADS: usize = 2;
    const NUM_TAKES: usize = 60;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let images = [
        imread(
            "tests/vision/resources/gate_images/straight_on_0.png",
            IMREAD_COLOR,
        )
        .unwrap(),
        imread(
            "tests/vision/resources/buoy_images/straight_on_0.png",
            IMREAD_COLOR,
        )
        .unwrap(),
    ];

    let models_gen: [Box<dyn Fn(usize, usize) -> ModelPipelined>; 2] = [
        Box::new(|model_threads, post_processing_threads| {
            runtime.block_on(GatePoles::default().into_pipelined(
                NonZeroUsize::try_from(model_threads).unwrap(),
                NonZeroUsize::try_from(post_processing_threads).unwrap(),
            ))
        }),
        Box::new(|model_threads, post_processing_threads| {
            runtime.block_on(BuoyModel::default().into_pipelined(
                NonZeroUsize::try_from(model_threads).unwrap(),
                NonZeroUsize::try_from(post_processing_threads).unwrap(),
            ))
        }),
    ];

    let group_names = ["gate_pipeline".to_string(), "buoy_pipeline".to_string()];

    group_names
        .into_iter()
        .zip(images)
        .zip(models_gen)
        .for_each(|((group_name, image), model_gen)| {
            let mut group = c.benchmark_group(group_name + " (" + CUDA_ENABLED + ")");
            group.sample_size(10);

            for model_threads in 1..=MAX_MODEL_THREADS {
                for post_processing_threads in 1..=MAX_POST_PROCESSING_THREADS {
                    let model = Arc::new(model_gen(model_threads, post_processing_threads));
                    let image = MatWrapper(image.clone());

                    // Approximate frame delivery from a 30 FPS camera
                    let model_clone = model.clone();
                    let feeder = runtime.spawn(async move {
                        loop {
                            model_clone.update_mat(image.clone());
                            sleep(Duration::from_secs_f64(1.0 / 30.0)).await;
                        }
                    });

                    group.bench_function(
                        BenchmarkId::from_parameter(format!(
                            "{:?}",
                            [model_threads, post_processing_threads]
                        )),
                        |b| {
                            b.to_async(&runtime).iter(|| async {
                                for _ in 0..NUM_TAKES {
                                    black_box(model.get_single().await);
                                }
                            })
                        },
                    );

                    feeder.abort();
                }
            }

            group.finish();
        });
    runtime.shutdown_background();
}

fn stages(c: &mut Criterion) {
    let images = [
        imread(
            "tests/vision/resources/buoy_images/straight_on_0.png",
            IMREAD_COLOR,
        )
        .unwrap(),
        imread(
            "tests/vision/resources/buoy_images/straight_on_0.png",
            IMREAD_COLOR,
        )
        .unwrap(),
    ];
    let models = [
        GatePoles::default().model().clone(),
        BuoyModel::default().model().clone(),
    ];
    let thresholds = [
        *GatePoles::default().threshold(),
        *BuoyModel::default().threshold(),
    ];

    let names = ["Gate", "Buoy"];

    images
        .into_iter()
        .zip(models)
        .zip(names)
        .zip(thresholds)
        .for_each(|(((image, mut model), name), threshold)| {
            c.bench_function(
                &(name.to_string() + " Forwarding (" + CUDA_ENABLED + ")"),
                |b| {
                    b.iter(|| {
                        black_box(model.forward(&image));
                    })
                },
            );

            let args = model.post_process_args();
            let forward_output = model.forward(&image);

            c.bench_function(
                &(name.to_string() + " Post Processing (" + CUDA_ENABLED + ")"),
                |b| {
                    b.iter(|| {
                        black_box(OnnxModel::post_process(
                            args,
                            forward_output.clone(),
                            threshold,
                        ));
                    })
                },
            );
        });
}

criterion_group!(model_processing, gate_pole_model, buoy_model);
criterion_group!(model_processing_throughput, stages, pipelined);
criterion_main!(model_processing, model_processing_throughput);
