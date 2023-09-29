// https://github.com/Nic-Gould/rusty-yolo/tree/main/models
// https://github.com/igor-yusupov/rusty-yolo
// https://github.com/LaurentMazare/tch-rs
// https://pytorch.org/get-started/locally/ # install LibTorch cxx11 ABI, 

// export LIBTORCH=/path/to/libtorch
// export LIBTORCH_INCLUDE=/path/to/libtorch/
// export LIBTORCH_LIB=/path/to/libtorch/


// OpenCV build after clean
// export VCPKGRS_DYNAMIC=1
// cargo clean and build until it works
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use rusty_yolo;
    use tch;
    #[test]
    fn inference() {
        // check cuda options (cpu or gpu)
        let device = tch::Device::cuda_if_available();

        // load converted model (torchscript without tuple, see https://github.com/Nic-Gould/rusty-yolo/tree/main/models for instruction)
        let model = rusty_yolo::YOLO::new("/home/lixin/Projects/RustProjects/SW8S-Rust/src/vision/models/yolo.torchscript", 320, 320, device);

        let mut orig_image = tch::vision::image::load("/home/lixin/Projects/RustProjects/SW8S-Rust/tests/vision/resources/buoy_images/1.jpeg").unwrap();

        let results = model.predict(&orig_image, 0.5, 0.35);    // inference (img, conf, iou)

        assert_approx_eq!(results.get(0).unwrap().xmin, 98.051513671875);

        model.draw_rectangle(&mut orig_image, &results);
        tch::vision::image::save(&orig_image, "/home/lixin/Projects/RustProjects/SW8S-Rust/tests/vision/output/buoy_images/torch_test.jpeg").expect("Failed to save");
    }
}