use super::action_context::{BottomCamIO, GetControlBoard, GetMainElectronicsBoard};
use crate::config::spin::Config;
use tokio::io::WriteHalf;
use tokio::time::{sleep, Duration};
use tokio_serial::SerialStream;

pub async fn spin<
    Con: Send + Sync + GetControlBoard<WriteHalf<SerialStream>> + GetMainElectronicsBoard + BottomCamIO,
>(
    context: &Con,
    config: &Config,
) {
    #[cfg(feature = "logging")]
    logln!("Starting spin");

    let cb = context.get_control_board();
    let _ = cb.bno055_periodic_read(true).await;

    let initial_roll = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.roll();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let initial_yaw = loop {
        if let Some(initial_angle) = cb.responses().get_angles().await {
            break *initial_angle.yaw();
        } else {
            #[cfg(feature = "logging")]
            logln!("Failed to get initial angle");
        }
    };

    let mut spin_count = 0;
    let mut in_spin = false;

    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;
    sleep(Duration::from_secs(1)).await;
    let _ = cb
        .global_speed_set(0.0, 0.0, 0.0, 0.0, config.spin_speed, 0.0)
        .await;

    loop {
        let curr_roll = loop {
            if let Some(angle) = cb.responses().get_angles().await {
                break *angle.roll();
            } else {
                #[cfg(feature = "logging")]
                logln!("Failed to get angle");
            }
        };

        let mut diff: f32 = curr_roll - initial_roll;
        if diff > 180.0 {
            diff -= 360.0;
        } else if diff < -180.0 {
            diff += 360.0;
        }

        if !in_spin && diff.abs() > config.hysteresis {
            in_spin = true;
        }

        if in_spin && diff.abs() < config.hysteresis {
            spin_count += 1;
            #[cfg(feature = "logging")]
            logln!("Completed spin {}", spin_count);
            in_spin = false;
        }

        if spin_count >= config.num_spins {
            break;
        }
    }
    let _ = cb
        .stability_2_speed_set(0.0, 0.0, 0.0, 0.0, initial_yaw, config.depth)
        .await;
}
