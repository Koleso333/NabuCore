use std::process::Stdio;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::stream;
use ncore::rt::{BoxFuture, BoxStream};
use ncore::{
    ChildProcess, ProcessControl, ProcessLauncher, ProcessSpec, ProcessStdin, TransportError,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{Notify, watch};

/// Launches children with `tokio::process`; must be called inside a tokio runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioLauncher;

impl ProcessLauncher for TokioLauncher {
    fn launch(&self, spec: &ProcessSpec) -> Result<ChildProcess, TransportError> {
        let mut cmd = Command::new(&spec.program);
        cmd.args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if spec.clear_env {
            cmd.env_clear();
        }
        cmd.envs(spec.env.iter().map(|(k, v)| (k, v)));
        if let Some(dir) = &spec.cwd {
            cmd.current_dir(dir);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| TransportError::connect(format!("cannot start {}: {e}", spec.program)))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");

        let (status_tx, status_rx) = watch::channel(None);
        let kill = Arc::new(Notify::new());
        let k = kill.clone();
        tokio::spawn(async move {
            let status = tokio::select! {
                s = child.wait() => s,
                _ = k.notified() => {
                    let _ = child.start_kill();
                    child.wait().await
                }
            };
            let code = status.ok().and_then(|s| s.code());
            let _ = status_tx.send(Some(code));
        });
        Ok(ChildProcess {
            stdin: Box::new(Stdin(Some(stdin))),
            stdout: chunks(stdout),
            stderr: chunks(stderr),
            control: Arc::new(Control {
                kill,
                status: status_rx,
            }),
        })
    }
}

fn chunks<R: AsyncRead + Unpin + Send + 'static>(
    r: R,
) -> BoxStream<'static, Result<Bytes, TransportError>> {
    Box::pin(stream::unfold(Some(r), |r| async move {
        let mut r = r?;
        let mut buf = vec![0u8; 64 * 1024];
        match r.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok(Bytes::from(buf)), Some(r)))
            }
            Err(e) => Some((Err(TransportError::new(e.to_string())), None)),
        }
    }))
}

struct Stdin(Option<ChildStdin>);

impl ProcessStdin for Stdin {
    fn write(&mut self, data: Bytes) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move {
            let s = self
                .0
                .as_mut()
                .ok_or_else(|| TransportError::new("stdin is closed"))?;
            s.write_all(&data)
                .await
                .map_err(|e| TransportError::new(e.to_string()))?;
            s.flush()
                .await
                .map_err(|e| TransportError::new(e.to_string()))
        })
    }

    fn close(&mut self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if let Some(mut s) = self.0.take() {
                let _ = s.shutdown().await;
            }
        })
    }
}

struct Control {
    kill: Arc<Notify>,
    status: watch::Receiver<Option<Option<i32>>>,
}

impl ProcessControl for Control {
    fn kill(&self) {
        self.kill.notify_one();
    }

    fn wait(&self) -> BoxFuture<'_, Option<i32>> {
        let mut rx = self.status.clone();
        Box::pin(async move {
            match rx.wait_for(Option::is_some).await {
                Ok(v) => v.flatten(),
                Err(_) => None,
            }
        })
    }
}
