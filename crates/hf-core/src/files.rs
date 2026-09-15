//! Exclusive file creation with the foundation's modes, the holdout refusal,
//! and Python-style pretty JSON (`json.dumps(indent=2, sort_keys=True,
//! ensure_ascii=False) + "\n"`, the foundation's `dump_pretty`).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use serde_json::Value;

use crate::HfError;

/// Visible artifacts.
pub const MODE_VISIBLE: u32 = 0o644;
/// Hidden streams and labels.
pub const MODE_HIDDEN: u32 = 0o600;
/// Split directories.
pub const MODE_SPLIT_DIR: u32 = 0o700;

/// Refuse any path naming a holdout, as every foundation CLI does.
pub fn refuse_holdout(path: &Path) -> Result<(), HfError> {
    let lower = path.to_string_lossy().to_lowercase();
    if lower.contains("holdout") || lower.contains("heldout") {
        return Err(HfError::Refused(format!(
            "holdout material is never touched: {}",
            path.display()
        )));
    }
    Ok(())
}

/// `os.open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode)`.
pub fn create_exclusive(path: &Path, mode: u32) -> std::io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc_nofollow())
        .mode(mode)
        .open(path)
}

fn libc_nofollow() -> i32 {
    // O_NOFOLLOW on Linux; kept as a literal so hf-core carries no libc dependency.
    0o400000
}

/// Write bytes to a new file exclusively, fsync, and return the path.
pub fn write_exclusive(path: &Path, mode: u32, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = create_exclusive(path, mode)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()
}

/// Python's `repr(float)`: shortest round-trip digits, exponent form outside
/// `1e-4 <= |x| < 1e16`, always a decimal point or an exponent.
pub fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let sci = format!("{:e}", x); // shortest round-trip, e.g. "2.5e-5", "1e21", "-1.5e0"
    let (mantissa, exponent) = sci.split_once('e').expect("scientific form");
    let exponent: i32 = exponent.parse().expect("exponent");
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-');
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        // positional
        let point = exponent + 1; // digits before the decimal point
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..(-point) {
                out.push('0');
            }
            out.push_str(&digits);
        } else if (point as usize) >= digits.len() {
            out.push_str(&digits);
            for _ in 0..(point as usize - digits.len()) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exponent.abs()));
    }
    out
}

fn python_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn python_json_value(value: &Value, indent: usize, depth: usize, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&python_float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => python_json_string(s, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.push_str(&" ".repeat(indent * (depth + 1)));
                python_json_value(item, indent, depth + 1, out);
            }
            out.push('\n');
            out.push_str(&" ".repeat(indent * depth));
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort(); // byte order == code-point order == Python's str order
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.push_str(&" ".repeat(indent * (depth + 1)));
                python_json_string(key, out);
                out.push_str(": ");
                python_json_value(&map[*key], indent, depth + 1, out);
            }
            out.push('\n');
            out.push_str(&" ".repeat(indent * depth));
            out.push('}');
        }
    }
}

/// `json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)`; no newline.
pub fn python_json_pretty(value: &Value) -> String {
    let mut out = String::new();
    python_json_value(value, 2, 0, &mut out);
    out
}

/// `dump_pretty`: the above plus a trailing newline — for humans, never hashed.
pub fn dump_pretty(value: &Value) -> String {
    python_json_pretty(value) + "\n"
}

/// `json.dumps(value)` with Python's default separators (`", "` and `": "`),
/// keys in insertion order, `ensure_ascii=False`; the sidecar line format.
pub fn python_json_compact(value: &Value) -> String {
    fn walk(value: &Value, out: &mut String) {
        match value {
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    walk(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    python_json_string(k, out);
                    out.push_str(": ");
                    walk(v, out);
                }
                out.push('}');
            }
            other => python_json_value(other, 0, 0, out),
        }
    }
    let mut out = String::new();
    walk(value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (100.0, "100.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (2.5e-5, "2.5e-05"),
            (1e21, "1e+21"),
            (-0.0, "-0.0"),
            (std::f64::consts::PI, "3.141592653589793"),
            (123456789.125, "123456789.125"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(python_float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn pretty_matches_python_layout() {
        let v = json!({"b": [1, 2], "a": {"y": "é\n", "x": []}, "c": {}, "d": 0.5});
        assert_eq!(
            dump_pretty(&v),
            "{\n  \"a\": {\n    \"x\": [],\n    \"y\": \"é\\n\"\n  },\n  \"b\": [\n    1,\n    2\n  ],\n  \"c\": {},\n  \"d\": 0.5\n}\n"
        );
        assert_eq!(
            python_json_compact(&json!({"node": "Q1", "vector": [0.5, 1.0]})),
            "{\"node\": \"Q1\", \"vector\": [0.5, 1.0]}"
        );
    }

    #[test]
    fn holdout_paths_are_refused() {
        assert!(refuse_holdout(Path::new("private/read-run-v1/HELDOUT.seed")).is_err());
        assert!(refuse_holdout(Path::new("private/real-walk-v1/splits/x")).is_ok());
    }

    #[test]
    fn exclusive_creation_refuses_an_existing_file() {
        let dir = std::env::temp_dir().join(format!("hf-core-files-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("once");
        write_exclusive(&path, MODE_HIDDEN, b"x").unwrap();
        assert!(write_exclusive(&path, MODE_HIDDEN, b"y").is_err());
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            MODE_HIDDEN
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
