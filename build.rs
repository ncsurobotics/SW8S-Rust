// fn main() {
//     // #[cfg(feature = "cuda")]
//     // {
//     //     // https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
//     //     const COMPUTE_CODES: &[&str] = &[
//     //         "52", "53", "60", "61", "62", "70", "72", "75", "80", "86", "87", "89", "90", "90a",
//     //     ];

//     //     // Rebuild on any kernel change
//     //     println!("cargo:rerun-if-changed=src/cuda_kernels");

//     //     // Rebuild for specific files that use kernels changing
//     //     println!("cargo:rerun-if-changed=src/vision/nn_cv2.rs");

//     //     let mut build = cc::Build::new();
//     //     build.cuda(true).flag("-cudart=shared");

//     //     for code in COMPUTE_CODES {
//     //         build
//     //             .flag("-gencode")
//     //             .flag(&format!("arch=compute_{},code=sm_{}", code, code));
//     //     }

//     //     println!("cargo:rustc-link-lib=cudart");
//     // }
// }
