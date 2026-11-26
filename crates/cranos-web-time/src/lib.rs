pub use core::time::Duration;
use core::ops::{Add, AddAssign, Sub, SubAssign};

#[cfg(target_os = "zkvm")]
unsafe extern "C" {
    fn cranos_clock_nanos() -> u64;
}

#[cfg(target_os = "zkvm")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(u64);

#[cfg(target_os = "zkvm")]
impl Instant {
    #[inline]
    pub fn now() -> Self {
        Self(unsafe { cranos_clock_nanos() })
    }

    #[inline]
    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.checked_duration_since(earlier)
            .expect("supplied instant is later than self")
    }

    #[inline]
    pub fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }

    #[inline]
    pub fn saturating_duration_since(&self, earlier: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        Self::now().saturating_duration_since(*self)
    }

    #[inline]
    pub fn checked_add(&self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;
        self.0.checked_add(nanos).map(Self)
    }

    #[inline]
    pub fn checked_sub(&self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;
        self.0.checked_sub(nanos).map(Self)
    }
}

#[cfg(target_os = "zkvm")]
impl Add<Duration> for Instant {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Duration) -> Self {
        Self(self.0.saturating_add(rhs.as_nanos().min(u64::MAX as u128) as u64))
    }
}

#[cfg(target_os = "zkvm")]
impl AddAssign<Duration> for Instant {
    #[inline]
    fn add_assign(&mut self, rhs: Duration) {
        *self = *self + rhs;
    }
}

#[cfg(target_os = "zkvm")]
impl Sub<Duration> for Instant {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Duration) -> Self {
        Self(self.0.saturating_sub(rhs.as_nanos().min(u64::MAX as u128) as u64))
    }
}

#[cfg(target_os = "zkvm")]
impl SubAssign<Duration> for Instant {
    #[inline]
    fn sub_assign(&mut self, rhs: Duration) {
        *self = *self - rhs;
    }
}

#[cfg(target_os = "zkvm")]
impl Sub<Instant> for Instant {
    type Output = Duration;
    #[inline]
    fn sub(self, rhs: Self) -> Duration {
        self.saturating_duration_since(rhs)
    }
}

#[cfg(target_os = "zkvm")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemTime(u64);

#[cfg(target_os = "zkvm")]
pub const UNIX_EPOCH: SystemTime = SystemTime(0);

#[cfg(target_os = "zkvm")]
#[derive(Clone, Debug)]
pub struct SystemTimeError(Duration);

#[cfg(target_os = "zkvm")]
impl SystemTimeError {
    pub fn duration(&self) -> Duration {
        self.0
    }
}

#[cfg(target_os = "zkvm")]
impl core::fmt::Display for SystemTimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "second time provided was later than self")
    }
}

#[cfg(target_os = "zkvm")]
impl std::error::Error for SystemTimeError {}

#[cfg(target_os = "zkvm")]
impl SystemTime {
    pub const UNIX_EPOCH: Self = UNIX_EPOCH;

    #[inline]
    pub fn now() -> Self {
        Self(unsafe { cranos_clock_nanos() })
    }

    pub fn duration_since(&self, earlier: Self) -> Result<Duration, SystemTimeError> {
        if self.0 >= earlier.0 {
            Ok(Duration::from_nanos(self.0 - earlier.0))
        } else {
            Err(SystemTimeError(Duration::from_nanos(earlier.0 - self.0)))
        }
    }

    pub fn elapsed(&self) -> Result<Duration, SystemTimeError> {
        Self::now().duration_since(*self)
    }
}

#[cfg(not(target_os = "zkvm"))]
pub use std::time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH};
