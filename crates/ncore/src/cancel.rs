use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

#[derive(Debug, Default)]
struct Inner {
    cancelled: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

/// Cloneable cancellation flag usable from any thread or FFI caller.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<Inner>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::SeqCst);
        for w in self.0.wakers.lock().unwrap().drain(..) {
            w.wake();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::SeqCst)
    }

    /// Resolves once `cancel` has been called.
    pub fn cancelled(&self) -> Cancelled {
        Cancelled(self.clone())
    }

    // Registers the waker; returns true if already cancelled.
    pub(crate) fn poll_cancelled(&self, cx: &mut Context<'_>) -> bool {
        if self.is_cancelled() {
            return true;
        }
        let mut wakers = self.0.wakers.lock().unwrap();
        if !wakers.iter().any(|w| w.will_wake(cx.waker())) {
            wakers.push(cx.waker().clone());
        }
        drop(wakers);
        self.is_cancelled()
    }
}

/// Future returned by `CancelToken::cancelled`.
#[derive(Debug)]
pub struct Cancelled(CancelToken);

impl Future for Cancelled {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0.poll_cancelled(cx) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Runs `fut` unless the token fires first.
pub async fn or_cancel<T>(cancel: &CancelToken, fut: impl Future<Output = T>) -> Option<T> {
    use futures_util::future::{Either, select};
    let fut = std::pin::pin!(fut);
    match select(fut, cancel.cancelled()).await {
        Either::Left((v, _)) => Some(v),
        Either::Right(_) => None,
    }
}
