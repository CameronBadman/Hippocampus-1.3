// Two jobs, both at compile time.
//
// 1. Build-time provenance. Every artifact used to carry `engine_head`, read by
//    running `git` in this checkout at RUN time — which says what the tree is at
//    now, not what the binary in hand was built from; a release binary built at
//    one commit happily stamped artifacts with a later one. The engine's HEAD,
//    whether the tree was dirty and when the build ran are embedded here instead,
//    as ENGINE_BUILD_HEAD / ENGINE_BUILD_DIRTY / ENGINE_BUILD_TIME, so the two
//    can be compared (and disagreement refused) at run time.
//
// 2. torch-sys emits the libtorch rpath only for its own targets, and links
//    `-ltorch_cuda` in a position where the linker's default --as-needed drops it
//    (nothing in the Rust code references a symbol from it, so the binary comes up
//    with CUDA unavailable). Both are fixed here for this crate's binaries: an rpath
//    to the LIBTORCH the build linked against (tools/env.sh sets it to the venv's
//    torch wheel), and an explicit --no-as-needed link of the CUDA libraries.

use std::path::{Path, PathBuf};

fn main() {
    // unconditionally, and before anything that may return early: a missing
    // `cargo:rustc-env` line is a compile error at the `env!` that reads it
    provenance();
    link_libtorch();
}

/// `git` in a repository, trimmed, or `None` when it fails or there is no git.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn provenance() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default()).join("../..");
    // A commit, a checkout or a branch move changes .git/HEAD or the ref it
    // names; an edit anywhere under crates/ changes the dirty flag. Both re-run
    // this script, so the embedded values follow the tree they were built from.
    // (An edit OUTSIDE crates/ — tools/, README — does not re-run it, so the
    // dirty flag is the tree as of the last build-script run; the head is not
    // affected, since a commit always moves HEAD.)
    let dotgit = root.join(".git");
    let head_file = dotgit.join("HEAD");
    if head_file.is_file() {
        println!("cargo:rerun-if-changed={}", head_file.display());
        if let Some(r) = std::fs::read_to_string(&head_file)
            .ok()
            .and_then(|t| t.trim().strip_prefix("ref: ").map(str::to_string))
        {
            let loose = dotgit.join(&r);
            let packed = dotgit.join("packed-refs");
            // a missing path counts as changed, so only watch one that is there
            if loose.is_file() {
                println!("cargo:rerun-if-changed={}", loose.display());
            } else if packed.is_file() {
                println!("cargo:rerun-if-changed={}", packed.display());
            }
        }
    }
    // crates/ and the manifests, never the workspace root: target/ churns
    for p in [root.join("crates"), root.join("Cargo.lock")] {
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
    let head = git(&root, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&root, &["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    println!("cargo:rustc-env=ENGINE_BUILD_HEAD={head}");
    println!("cargo:rustc-env=ENGINE_BUILD_DIRTY={dirty}");
    println!(
        "cargo:rustc-env=ENGINE_BUILD_TIME={}",
        hf_core::utc_now_iso()
    );
}

fn link_libtorch() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_LIB");
    let lib = match (std::env::var("LIBTORCH_LIB"), std::env::var("LIBTORCH")) {
        (Ok(lib), _) => lib,
        (_, Ok(root)) => format!("{root}/lib"),
        _ => return,
    };
    // binaries, integration tests and examples of this crate all link libtorch
    for kind in ["bins", "tests"] {
        println!("cargo:rustc-link-arg-{kind}=-Wl,-rpath,{lib}");
        println!("cargo:rustc-link-arg-{kind}=-L{lib}");
        println!("cargo:rustc-link-arg-{kind}=-Wl,--no-as-needed");
        for name in ["torch_cuda", "c10_cuda"] {
            if Path::new(&format!("{lib}/lib{name}.so")).exists() {
                println!("cargo:rustc-link-arg-{kind}=-l{name}");
            }
        }
        println!("cargo:rustc-link-arg-{kind}=-Wl,--as-needed");
    }
}
