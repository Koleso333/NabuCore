use std::time::Duration;

use ncore_core::ErrorKind;

/// Host-supplied retry policy; without one, failures go straight to the host.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Total attempts including the first one.
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub respect_retry_after: bool,
    pub retry_connect_errors: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            respect_retry_after: true,
            retry_connect_errors: true,
        }
    }
}

impl RetryPolicy {
    /// Whether an HTTP status is worth retrying.
    pub fn retries_status(&self, status: u16) -> bool {
        ErrorKind::from_status(status).is_transient()
    }

    /// Delay before attempt `attempt + 1`, where `attempt` starts at 1.
    pub fn delay(&self, attempt: u32, retry_after: Option<&str>) -> Duration {
        if self.respect_retry_after
            && let Some(secs) = retry_after.and_then(|v| v.trim().parse::<f64>().ok())
            && secs.is_finite()
            && secs >= 0.0
        {
            return Duration::from_secs_f64(secs).min(self.max_delay);
        }
        let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_and_retry_after() {
        let p = RetryPolicy::default();
        assert_eq!(p.delay(1, None), Duration::from_millis(500));
        assert_eq!(p.delay(3, None), Duration::from_secs(2));
        assert_eq!(p.delay(20, None), Duration::from_secs(30));
        assert_eq!(p.delay(1, Some("2")), Duration::from_secs(2));
        assert!(p.retries_status(529) && p.retries_status(429) && !p.retries_status(400));
    }
}
