use anyhow::Result;
use std::time::Duration;
use std::{fs::create_dir_all, path::Path};
use sw8s_rust_lib::comms::control_board::ControlBoard;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::sleep;

#[cfg(target_os = "linux")]
use {flate2::bufread::GzDecoder, tar::Archive};

#[cfg(target_os = "linux")]
static GODOT: Mutex<&str> = Mutex::const_new("tests/godot_sim/GodotAUVSim.x86_64");
#[cfg(target_os = "macos")]
const GODOT: &str = "tests/godot_sim/GodotAUVSim.app";
#[cfg(target_os = "windows")]
const GODOT: &str = "tests/godot_sim/GodotAUVSim.exe";

const GODOT_DIR: &str = "tests/godot_sim/";

async fn download_sim() -> Result<()> {
    const VERSION: &str = "v1.2.1";

    #[cfg(target_os = "linux")]
    const SUFFIX: &str = "Linux_amd64.tar.gz";
    #[cfg(target_os = "macos")]
    const SUFFIX: &str = "macOS_amd64.app.zip";
    #[cfg(target_os = "windows")]
    const SUFFIX: &str = "Windows_amd64.zip";

    create_dir_all(GODOT_DIR)?;
    let source = format!("https://github.com/MB3hel/GodotAUVSim/releases/download/{VERSION}/GodotAUVSim_{VERSION}_{SUFFIX}");
    let response: &[u8] = &reqwest::get(source).await?.bytes().await?;

    #[cfg(target_os = "linux")]
    Archive::new(GzDecoder::new(response)).unpack(GODOT_DIR)?;

    #[cfg(not(target_os = "linux"))]
    zip_extract::extract(response, GODOT_DIR, true)?;

    Ok(())
}

async fn open_sim(godot: String) -> Result<()> {
    if !Path::new(&godot).is_file() {
        download_sim().await?
    }

    tokio::spawn(async move {
        Command::new(godot)
            .arg("--simcb")
            .kill_on_drop(true)
            .spawn()
            .unwrap()
            .wait()
            .await
            .unwrap()
    });
    // Give simulator time to spawn, magic number
    sleep(Duration::from_secs(3)).await;
    Ok(())
}

#[tokio::test]
pub async fn tcp_connect() {
    const LOCALHOST: &str = "127.0.0.1";
    const SIM_PORT: &str = "5011";

    open_sim(GODOT.lock().await.to_string()).await.unwrap();
    let _ = ControlBoard::tcp(LOCALHOST, SIM_PORT).await.unwrap();
}
