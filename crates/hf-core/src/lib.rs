//! Shared discipline for the real-walk engine.
//!
//! Every binary in this workspace reports a governance failure the way the
//! Python packages do: one line `<prog>: <message>` on stderr and exit code 2,
//! with any report it was writing still written. Nothing here trains.

pub mod canonical;
pub mod files;
pub mod pyrandom;

use std::fmt;

pub use canonical::{canonical_bytes, canonical_sha256, sha256_bytes, sha256_file};
pub use files::{dump_pretty, python_json_compact, refuse_holdout, write_exclusive};
pub use pyrandom::{PyRandom, PyRandomState};

/// A refusal the operator must read: the artifact is wrong, not the code.
#[derive(Debug, thiserror::Error)]
pub enum HfError {
    /// Band H: an integrity gate failed; nothing downstream may be read.
    #[error("band H: {0}")]
    BandH(String),
    /// The command was asked to do something the rule does not authorise.
    #[error("refusing: {0}")]
    Refused(String),
    /// The input is malformed or missing.
    #[error("{0}")]
    Invalid(String),
}

/// The process exit code every governance failure maps to, as in the Python CLIs.
pub const GOVERNANCE_EXIT_CODE: i32 = 2;

/// Print `<prog>: <message>` and exit 2; the convention shared with `hf-phase0` etc.
pub fn exit_with(prog: &str, error: &dyn fmt::Display) -> ! {
    eprintln!("{prog}: {error}");
    std::process::exit(GOVERNANCE_EXIT_CODE)
}

/// ISO-8601 UTC at seconds resolution, as the runner stamps `started_at`.
pub fn utc_now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // civil-from-days (Howard Hinnant), avoiding a time dependency in hf-core
    let days = (now / 86_400) as i64;
    let secs = now % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}
