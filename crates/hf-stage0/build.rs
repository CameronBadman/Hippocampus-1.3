// torch-sys emits the libtorch rpath only for its own targets, and links
// `-ltorch_cuda` in a position where the linker's default --as-needed drops it
// (nothing in the Rust code references a symbol from it, so the binary comes up
// with CUDA unavailable). Both are fixed here for this crate's binaries: an rpath
// to the LIBTORCH the build linked against (tools/env.sh sets it to the venv's
// torch wheel), and an explicit --no-as-needed link of the CUDA libraries.
fn main() {
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
            if std::path::Path::new(&format!("{lib}/lib{name}.so")).exists() {
                println!("cargo:rustc-link-arg-{kind}=-l{name}");
            }
        }
        println!("cargo:rustc-link-arg-{kind}=-Wl,--as-needed");
    }
}
