use core::fmt::Debug;
use std::{ops::Deref, sync::Arc, time::Duration};

use anyhow::{bail, Result};
use tokio::{
    io::{self, AsyncRead, AsyncWrite, AsyncWriteExt, WriteHalf},
    net::TcpStream,
    sync::Mutex,
    time::{sleep, timeout},
};
use tokio_serial::{DataBits, Parity, SerialStream, StopBits};

use self::{response::ResponseMap, util::BNO055AxisConfig};

use super::auv_control_board::{AUVControlBoard, MessageId};

pub mod response;
pub mod util;

#[derive(Debug)]
pub struct ControlBoard<T>
where
    T: AsyncWriteExt + Unpin,
{
    inner: Arc<AUVControlBoard<T, ResponseMap>>,
}

impl<T: AsyncWriteExt + Unpin> Deref for ControlBoard<T> {
    type Target = AUVControlBoard<T, ResponseMap>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: 'static + AsyncWriteExt + Unpin + Send> ControlBoard<T> {
    pub async fn new<U>(comm_out: T, comm_in: U, msg_id: Option<MessageId>) -> Result<Self>
    where
        U: 'static + AsyncRead + Unpin + Send,
    {
        const THRUSTER_INVS: [bool; 8] = [true, true, false, false, true, false, false, true];
        #[allow(clippy::approx_constant)]
        const DOF_SPEEDS: [f32; 6] = [0.7071, 0.7071, 1.0, 0.4413, 1.0, 0.8139];

        let msg_id = msg_id.unwrap_or_default();
        let responses = ResponseMap::new(comm_in).await;
        let this = Self {
            inner: AUVControlBoard::new(Mutex::from(comm_out).into(), responses, msg_id).into(),
        };

        this.init_matrices().await?;
        this.thruster_inversion_set(&THRUSTER_INVS).await?;
        this.relative_dof_speed_set_batch(&DOF_SPEEDS).await?;
        this.bno055_imu_axis_config(BNO055AxisConfig::P6).await?;

        loop {
            if let Ok(ret) = timeout(Duration::from_secs(1), this.raw_speed_set([0.0; 8])).await {
                ret?;
                break;
            }
        }

        // Control board needs time to get its life together
        sleep(Duration::from_secs(5)).await;

        this.stab_tune().await?;

        let inner_clone = this.inner.clone();

        tokio::spawn(async move {
            loop {
                if let Ok(inner) = timeout(
                    Duration::from_millis(100),
                    Self::feed_watchdog(&inner_clone),
                )
                .await
                {
                    inner.unwrap();
                } else {
                    eprintln!("Watchdog ACK timed out.");
                }

                sleep(Duration::from_millis(200)).await;
            }
        });

        // Wait for watchdog to register
        while this.watchdog_status().await != Some(true) {
            sleep(Duration::from_millis(10)).await;
        }
        Ok(this)
    }

    async fn init_matrices(&self) -> Result<()> {
        self.motor_matrix_set(3, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0)
            .await?;
        self.motor_matrix_set(4, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0)
            .await?;
        self.motor_matrix_set(1, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0)
            .await?;
        self.motor_matrix_set(2, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            .await?;
        self.motor_matrix_set(7, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0)
            .await?;
        self.motor_matrix_set(8, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0)
            .await?;
        self.motor_matrix_set(5, 0.0, 0.0, -1.0, 1.0, -1.0, 0.0)
            .await?;
        self.motor_matrix_set(6, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0)
            .await?;

        self.motor_matrix_update().await
    }

    async fn stab_tune(&self) -> Result<()> {
        self.stability_assist_pid_tune('X', 0.8, 0.0, 0.0, 0.6, false)
            .await?;
        self.stability_assist_pid_tune('Y', 0.15, 0.0, 0.0, 0.1, false)
            .await?;
        self.stability_assist_pid_tune('Z', 1.6, 1e-6, 0.0, 0.8, false)
            .await?;
        self.stability_assist_pid_tune('D', 1.5, 0.0, 0.0, 1.0, false)
            .await
    }
}

impl ControlBoard<WriteHalf<SerialStream>> {
    pub async fn serial(port_name: &str) -> Result<Self> {
        const BAUD_RATE: u32 = 9600;
        const DATA_BITS: DataBits = DataBits::Eight;
        const PARITY: Parity = Parity::None;
        const STOP_BITS: StopBits = StopBits::One;

        let port_builder = tokio_serial::new(port_name, BAUD_RATE)
            .data_bits(DATA_BITS)
            .parity(PARITY)
            .stop_bits(STOP_BITS);
        let (comm_in, comm_out) = io::split(SerialStream::open(&port_builder)?);
        Self::new(comm_out, comm_in, None).await
    }
}

impl ControlBoard<WriteHalf<TcpStream>> {
    /// Both connections are necessary for the simulator to run,
    /// but the one that doesn't feed forward to control board is unnecessary
    pub async fn tcp(host: &str, port: &str, dummy_port: String) -> Result<Self> {
        let host = host.to_string();
        let host_clone = host.clone();
        tokio::spawn(async move {
            let _stream = TcpStream::connect(host_clone + ":" + &dummy_port)
                .await
                .unwrap();
            // Have to avoid dropping the TCP stream
            loop {
                sleep(Duration::MAX).await
            }
        });

        let stream = TcpStream::connect(host.to_string() + ":" + port).await?;
        let (comm_in, comm_out) = io::split(stream);
        Self::new(comm_out, comm_in, None).await
    }
}

impl<T: AsyncWrite + Unpin> ControlBoard<T> {
    pub async fn feed_watchdog(control_board: &Arc<AUVControlBoard<T, ResponseMap>>) -> Result<()> {
        const WATCHDOG_FEED: [u8; 4] = *b"WDGF";
        let message = Vec::from(WATCHDOG_FEED);
        control_board.write_out_basic(message).await
    }

    /// https://mb3hel.github.io/AUVControlBoard/user_guide/messages/#configuration-commands
    #[allow(clippy::too_many_arguments)]
    pub async fn motor_matrix_set(
        &self,
        thruster: u8,
        x: f32,
        y: f32,
        z: f32,
        pitch: f32,
        roll: f32,
        yaw: f32,
    ) -> Result<()> {
        const MOTOR_MATRIX_SET: [u8; 5] = *b"MMATS";
        // Oversized to avoid reallocations
        let mut message: Vec<u8> = Vec::with_capacity(32 * 8);
        message.extend(MOTOR_MATRIX_SET);

        if !(1..=8).contains(&thruster) {
            bail!("{thruster} is outside the allowed range 1-8.")
        };

        message.extend(thruster.to_le_bytes());
        [x, y, z, pitch, roll, yaw]
            .iter()
            .for_each(|val| message.extend(val.to_le_bytes()));

        self.write_out_basic(message).await
    }

    pub async fn motor_matrix_update(&self) -> Result<()> {
        const MOTOR_MATRIX_UPDATE: [u8; 5] = *b"MMATU";
        self.write_out_basic(Vec::from(MOTOR_MATRIX_UPDATE)).await
    }

    /// Set thruster inversions
    ///
    /// # Arguments:
    /// * `inversions` - Array of invert statuses, with motor 1 at index 0
    pub async fn thruster_inversion_set(&self, inversions: &[bool; 8]) -> Result<()> {
        const THRUSTER_INVERSION_SET: [u8; 4] = *b"TINV";
        let mut message = Vec::from(THRUSTER_INVERSION_SET);

        // bitmask to u8, may or may not outpreform mutating a single u8
        message.push(
            inversions
                .iter()
                .enumerate()
                .map(|(idx, &inv)| (inv as u8) << idx)
                .sum(),
        );
        self.write_out_basic(message).await
    }

    pub async fn relative_dof_speed_set(
        &self,
        x: f32,
        y: f32,
        z: f32,
        xrot: f32,
        yrot: f32,
        zrot: f32,
    ) -> Result<()> {
        self.relative_dof_speed_set_batch(&[x, y, z, xrot, yrot, zrot])
            .await
    }

    pub async fn relative_dof_speed_set_batch(&self, values: &[f32; 6]) -> Result<()> {
        const DOF_SET: [u8; 6] = *b"RELDOF";
        // Oversized to avoid reallocations
        let mut message = Vec::with_capacity(32 * 8);
        message.extend(DOF_SET);

        values
            .iter()
            .for_each(|val| message.extend(val.to_le_bytes()));

        self.write_out_basic(message).await
    }

    pub async fn raw_speed_set(&self, speeds: [f32; 8]) -> Result<()> {
        const RAW_SET: [u8; 3] = *b"RAW";
        // Oversized to avoid reallocations
        let mut message = Vec::with_capacity(32 * 8);
        message.extend(RAW_SET);

        speeds
            .iter()
            .for_each(|val| message.extend(val.to_le_bytes()));

        self.write_out_basic(message).await
    }

    pub async fn stability_2_speed_set(
        &self,
        x: f32,
        y: f32,
        target_pitch: f32,
        target_roll: f32,
        target_yaw: f32,
        target_depth: f32,
    ) -> Result<()> {
        const SASSIST_2: [u8; 8] = *b"SASSIST2";
        // Oversized to avoid reallocations
        let mut message = Vec::with_capacity(32 * 8);
        message.extend(SASSIST_2);

        [x, y, target_pitch, target_roll, target_yaw, target_depth]
            .iter()
            .for_each(|val| message.extend(val.to_le_bytes()));

        self.write_out_basic(message).await
    }

    pub async fn stability_1_speed_set(
        &self,
        x: f32,
        y: f32,
        yaw_speed: f32,
        target_pitch: f32,
        target_roll: f32,
        target_depth: f32,
    ) -> Result<()> {
        const SASSIST_2: [u8; 8] = *b"SASSIST1";
        // Oversized to avoid reallocations
        let mut message = Vec::with_capacity(32 * 8);
        message.extend(SASSIST_2);

        [x, y, yaw_speed, target_pitch, target_roll, target_depth]
            .iter()
            .for_each(|val| message.extend(val.to_le_bytes()));

        self.write_out_basic(message).await
    }

    pub async fn bno055_imu_axis_config(&self, config: BNO055AxisConfig) -> Result<()> {
        const BNO055A_CONFIG: [u8; 7] = *b"BNO055A";

        let mut message = Vec::from(BNO055A_CONFIG);
        message.push(config.into());

        self.write_out_basic(message).await
    }

    pub async fn stability_assist_pid_tune(
        &self,
        which: char,
        kp: f32,
        ki: f32,
        kd: f32,
        limit: f32,
        invert: bool,
    ) -> Result<()> {
        const STAB_TUNE: [u8; 9] = *b"SASSISTTN";
        // Oversized to avoid reallocations
        let mut message = Vec::with_capacity(32 * 8);
        message.extend(STAB_TUNE);

        if !['X', 'Y', 'Z', 'D'].contains(&which) {
            bail!("{which} is not a valid PID tune, pick from [X, Y, Z, D]")
        }
        message.push(which as u8);

        [kp, ki, kd, limit]
            .iter()
            .for_each(|val| message.extend(val.to_le_bytes()));
        message.push(invert as u8);

        self.write_out_basic(message).await
    }

    pub async fn watchdog_status(&self) -> Option<bool> {
        *self.responses().watchdog_status().read().await
    }
}
