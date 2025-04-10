/// Adapted from a script written by Marcus Behel
use std::{
    env::{args, current_dir, set_var},
    fmt::Write,
    fs::read_to_string,
    process::{exit, Command},
    thread,
};

use anyhow::{anyhow, Context, Result};
use futures_util::TryStreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use serde::Deserialize;
use tar::Archive;
use tokio::{spawn, task::spawn_blocking};
use tokio_util::io::{StreamReader, SyncIoBridge};
use walkdir::WalkDir;
use which::which;
use xz::read::XzDecoder;

#[tokio::main]
async fn main() -> Result<()> {
    println!("This tool is run to build SW8S-Rust for the Jetson Nano.");
    println!("It downloads the \"sysroot-jetson\" subdirectory for libraries.");
    println!("It builds a binary in the \"jetson-target\" subdirectory.");
    println!("The default cargo command is a release \"build\" with cuda and logging, but arguments will override this command.");
    println!();

    let config_str =
        read_to_string("config.toml").context("Could not read config file config.toml")?;
    let config: Config = toml::from_str(config_str.as_str()).context("Failed to parse config")?;

    tools_check()?;

    let mut system_args = args().skip(1).collect::<Vec<_>>();
    if system_args.is_empty() {
        system_args = vec![
            "build".to_string(),
            "--release".to_string(),
            "--features".to_string(),
            "logging,annotated_streams".to_string(),
        ];
    }

    let cur_dir = current_dir().context("failure getting current directory")?;
    let parent_dir = cur_dir
        .parent()
        .ok_or_else(|| anyhow!("Failed to get parent of current directory"))?
        .canonicalize()?;
    let sysroot = parent_dir.join("sysroot-jetson");
    let sysroot_str = sysroot
        .to_str()
        .ok_or_else(|| anyhow!("Failed to stringify sysroot directory"))?;

    let multibar = MultiProgress::new();
    let multibar_clone = multibar.clone();

    // Jetson Nano architecture
    let toolchain_install = spawn_blocking(move || {
        // Prevent progress bars from overlapping with toolchain output
        multibar.set_draw_target(ProgressDrawTarget::hidden());

        Command::new("rustup")
            .args(["target", "add", "aarch64-unknown-linux-gnu"])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        multibar.set_draw_target(ProgressDrawTarget::stdout());
    });

    let sysroot_clone = sysroot.clone();
    let config_clone = config.clone();
    let get_sysroot = spawn(async {
        let sysroot = sysroot_clone;
        let config = config_clone;
        let multibar = multibar_clone;

        println!("Testing for sysroot");
        let need_sysroot;
        let sysroot_missing = !sysroot.exists();
        if let Some(fetch) = config.fetch_sysroot.to_owned() {
            need_sysroot = fetch && sysroot_missing;
        } else {
            need_sysroot = sysroot_missing;
        }
        if need_sysroot {
            // Streaming this process reduces I/O and reduces delay
            println!("Downloading sysroot...");

            let source = reqwest::get(config.sysroot_url).await.unwrap();

            multibar.set_move_cursor(true); // Reduce flickering
            let dl_bar = multibar.add(ProgressBar::new(source.content_length().unwrap_or(0)));
            // https://github.com/console-rs/indicatif/blob/main/examples/download.rs
            dl_bar.set_style(ProgressStyle::with_template("Download Progress: [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})").unwrap().with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));
            let xz_bar = multibar.add(ProgressBar::new(source.content_length().unwrap_or(0)));
            // https://github.com/console-rs/indicatif/blob/main/examples/download.rs
            xz_bar.set_style(
                ProgressStyle::with_template("Decompression: [{elapsed_precise}] {bytes}").unwrap(),
            );

            // Stream the download body
            let tarball_stream = dl_bar.wrap_async_read(StreamReader::new(
                source
                    .bytes_stream()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)),
            ));
            // Convert async IO to sync IO to do live XZ decoding
            let decoded_tarball = xz_bar.wrap_read(XzDecoder::new_multi_decoder(
                SyncIoBridge::new(tarball_stream),
            ));
            // Write out the tarball
            thread::spawn(|| Archive::new(decoded_tarball).unpack(sysroot).unwrap())
                .join()
                .unwrap();
            println!("Downloaded sysroot");
        } else {
            if sysroot_missing {
                eprintln!("Sysroot not found, fetching it is disabled");
                exit(1);
            } else {
                println!("Found sysroot");
            }
        }
    });

    // Passed to everything (c, c++, linker)
    let shared_flags = "-target aarch64-linux-gnu -mcpu=cortex-a57 -fuse-ld=lld --sysroot="
        .to_string()
        + sysroot_str
        + " -L"
        + sysroot_str
        + if cfg!(feature = "ubuntu") {
            "/usr/include -L"
        } else {
            "/usr/local/cuda-10.2/targets/aarch64-linux/lib/ -L"
        }
        + sysroot_str
        + "/usr/lib/aarch64-linux-gnu/";
    // Only to clang to compile C code
    let cflags = &shared_flags;
    // Only to clang++ to compile C++ code
    let cxxflags = &shared_flags;
    // To linker (and rustflags as link-args)
    let ldflags = &shared_flags;

    /*
     * Make sure any C/C++ code built by crates uses right compilers / flags
     * Note: Using triple specific vars so that tools built for build system as a
     * part of the build process build as intended.
     * Note that these should have target triple lower case unlike vars for cargo
     */
    set_var("CC_aarch64_unknown_linux_gnu", "clang");
    set_var("CXX_aarch64_unknown_linux_gnu", "clang++");
    set_var("AR_aarch64_unknown_linux_gnu", "llvm-ar");
    set_var("CFLAGS_aarch64_unknown_linux_gnu", cflags);
    set_var("CXXFLAGS_aarch64_unknown_linux_gnu", cxxflags);
    set_var("LDFLAGS_aarch64_unknown_linux_gnu", ldflags);

    // Cargo flags / tools setup for target
    set_var("CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER", "clang");
    set_var("CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_AR", "llvm-ar");
    let rustflags: String = ldflags
        .split_whitespace()
        .map(|arg| "-C link-args=".to_string() + arg + " ")
        .collect();
    set_var(
        "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
        rustflags,
    );

    set_var(
        "OPENCV_DISABLE_PROBES",
        "pkg_config,cmake,vcpkg_cmake,vcpkg",
    );

    // Need sysroot fully downloaded to system to search
    get_sysroot.await.unwrap();

    // OpenCV setup
    let opencv_link_libs: String = WalkDir::new(sysroot.join("./usr/lib/aarch64-linux-gnu/"))
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|f| f.file_name().to_string_lossy().to_string())
        .filter(|f| f.ends_with(".so") && f.starts_with("lib"))
        .map(|f| ",".to_string() + &f[3..f.len() - 3])
        .collect(); // remove beginning "lib" and ending ".so"
    set_var("OPENCV_LINK_LIBS", opencv_link_libs);
    set_var(
        "OPENCV_LINK_PATHS",
        sysroot.join("./usr/lib/aarch64-linux-gnu/"),
    );
    set_var(
        "OPENCV_INCLUDE_PATHS",
        sysroot
            .join("./usr/include/opencv4")
            .to_str()
            .unwrap()
            .to_string()
            + ","
            + sysroot
                .join("./usr/include/opencv4/opencv2")
                .to_str()
                .unwrap(),
    );

    // Wait for Jetson Nano toolchain
    toolchain_install
        .await
        .context("failure while waiting for Jetson Nano toolchain install")?;

    Command::new("cargo")
        .current_dir(parent_dir.clone())
        .args(system_args)
        .args([
            "--target",
            "aarch64-unknown-linux-gnu",
            "--target-dir",
            "target-jetson",
        ])
        .spawn().context("failure spawning cargo sub proccess")?
        .wait()
        .map_err(|e| anyhow!("Make sure current directory ({:?}) is the \"jetson\" subdirectory (SW8S-Rust/jetson)\n{:#?}", cur_dir, e))?;
    println!(
        "\nThe cross-compiled binary is in {:?}",
        parent_dir
            .join("target-jetson")
            .join("aarch64-unknown-linux-gnu")
    );
    Ok(())
}

/// Checks that all required programs are installed
fn tools_check() -> Result<()> {
    ["rustup", "cargo", "clang", "lld"]
        .into_iter()
        .try_for_each(program_check)
}

/// Checks that all programs are installed
fn program_check(program: &str) -> Result<()> {
    which(program).map_err(|_| anyhow!("{program} is not installed"))?;
    Ok(())
}

#[derive(Debug, Deserialize, Clone)]
struct Config {
    fetch_sysroot: Option<bool>,
    sysroot_url: String,
}
