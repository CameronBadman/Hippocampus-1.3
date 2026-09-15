//! Canonical JSON (RFC 8785, JCS) and the `sha256:<hex>` digest convention of
//! `hippocampus_foundation.phase0.canonical`, reproduced byte for byte.
//!
//! The Python side uses the `rfc8785` package, which refuses integers outside
//! the IEEE-754 safe domain and non-finite floats; the same refusals apply
//! here so both implementations either agree or both refuse.

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::HfError;

/// The largest integer magnitude JCS may carry (2^53 − 1), as `rfc8785` enforces.
pub const MAX_SAFE_INTEGER: i128 = (1i128 << 53) - 1;

fn check_domain(value: &Value) -> Result<(), HfError> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if (i as i128).abs() > MAX_SAFE_INTEGER {
                    return Err(HfError::Invalid(format!(
                        "{i} exceeds safe integer domain for JSON floats"
                    )));
                }
            } else if let Some(u) = n.as_u64() {
                if (u as i128) > MAX_SAFE_INTEGER {
                    return Err(HfError::Invalid(format!(
                        "{u} exceeds safe integer domain for JSON floats"
                    )));
                }
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return Err(HfError::Invalid(format!("{f} is not a finite float")));
                }
            }
            Ok(())
        }
        Value::Array(items) => items.iter().try_for_each(check_domain),
        Value::Object(map) => map.values().try_for_each(check_domain),
        _ => Ok(()),
    }
}

/// RFC 8785 bytes of a JSON value; no trailing newline, exactly as `canonical_bytes`.
pub fn canonical_bytes(value: &Value) -> Result<Vec<u8>, HfError> {
    check_domain(value)?;
    serde_json_canonicalizer::to_vec(value)
        .map_err(|e| HfError::Invalid(format!("canonical JSON: {e}")))
}

/// `sha256:<hex>` over arbitrary bytes.
pub fn sha256_bytes(data: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(data)))
}

/// `sha256:<hex>` over a value's canonical bytes — `canonical_sha256` in Python.
pub fn canonical_sha256(value: &Value) -> Result<String, HfError> {
    Ok(sha256_bytes(&canonical_bytes(value)?))
}

/// A streaming SHA-256 over a file, returning `(size, "sha256:<hex>")` like
/// the Python `_sha256_file`.
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<(u64, String)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        size += n as u64;
    }
    Ok((size, format!("sha256:{}", hex::encode(hasher.finalize()))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn digest_has_the_python_prefix_and_is_stable() {
        assert_eq!(
            sha256_bytes(b""),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn keys_are_sorted_and_floats_are_es6() {
        let bytes = canonical_bytes(&json!({"b": 1.0, "a": [1e21, 0.5]})).unwrap();
        assert_eq!(bytes, br#"{"a":[1e+21,0.5],"b":1}"#);
    }

    #[test]
    fn unsafe_integers_and_non_finite_floats_are_refused() {
        assert!(canonical_bytes(&json!({"n": 9007199254740992i64})).is_err());
        assert!(canonical_bytes(&json!({"n": -9007199254740992i64})).is_err());
        assert!(canonical_bytes(&json!({"n": 9007199254740991i64})).is_ok());
    }
}
