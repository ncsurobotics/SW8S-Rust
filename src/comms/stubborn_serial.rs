use std::ops::{Deref, DerefMut};

use futures::Future;
use stubborn_io::tokio::{StubbornIo, UnderlyingIo};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_serial::{SerialPortBuilder, SerialStream};

#[pin_project::pin_project]
pub struct SerialStreamWrapper(#[pin] pub SerialStream);

impl Deref for SerialStreamWrapper {
    type Target = SerialStream;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SerialStreamWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsyncRead for SerialStreamWrapper {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut ReadBuf,
    ) -> std::task::Poll<tokio::io::Result<()>> {
        self.project().0.poll_read(cx, buf)
    }
}

impl AsyncWrite for SerialStreamWrapper {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        self.project().0.poll_write(cx, buf)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.project().0.poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        self.project().0.poll_shutdown(cx)
    }
}

impl UnderlyingIo<SerialPortBuilder> for SerialStreamWrapper {
    fn establish(
        settings: SerialPortBuilder,
    ) -> std::pin::Pin<Box<dyn Future<Output = std::io::Result<Self>> + Send>> {
        Box::pin(async move { Ok(SerialStreamWrapper(SerialStream::open(&settings)?)) })
    }
}

pub type StubbornSerialStream = StubbornIo<SerialStreamWrapper, SerialPortBuilder>;
