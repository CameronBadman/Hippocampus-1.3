//! Step 0's toolchain gate: prove tch-rs links the venv's libtorch, sees the
//! GPU, runs a matmul and a backward on it, and reports what it linked.
use tch::{Device, Kind, Tensor};

fn main() -> anyhow::Result<()> {
    println!("cuda available: {}", tch::Cuda::is_available());
    println!("cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("cuda device count: {}", tch::Cuda::device_count());
    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let a = Tensor::randn([512, 512], (Kind::Float, device)).set_requires_grad(true);
    let b = Tensor::randn([512, 512], (Kind::Float, device));
    let c = a.matmul(&b);
    let loss = c.pow_tensor_scalar(2).mean(Kind::Float);
    loss.backward();
    let grad = a.grad();
    println!(
        "device {:?}: loss {:.4}, grad norm {:.4}, grad shape {:?}",
        device,
        f64::try_from(&loss)?,
        f64::try_from(grad.norm())?,
        grad.size()
    );
    let smoke_ok = f64::try_from(grad.norm())?.is_finite();
    println!("smoke {}", if smoke_ok { "OK" } else { "FAILED" });
    if !smoke_ok {
        std::process::exit(1);
    }
    Ok(())
}
