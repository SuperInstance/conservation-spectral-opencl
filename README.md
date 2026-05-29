# Conservation Spectral SDK — OpenCL

**Cross-Vendor GPU Compute for Spectral Graph Analysis**

The OpenCL implementation of the Conservation Spectral framework. Laplacian construction, power-iteration eigendecomposition, conservation ratios, Fiedler vector, and anomaly detection — all accelerated on GPU. Runs on NVIDIA, AMD, Intel, and Apple Silicon via OpenCL.

## The Aha Moment

OpenCL is the *United Nations of GPU compute*. It doesn't care if you're running NVIDIA, AMD, Intel, or Apple — the same kernels compile and run everywhere. This matters for conservation analysis because the real world doesn't have homogeneous hardware. A researcher analyzing food web conservation on their MacBook (Apple Silicon), a data center running AMD MI300X, and a workstation with an RTX 4090 should all get the same answers. OpenCL guarantees that. The `laplacian`, `power_iteration`, `conservation`, `fiedler`, and `anomaly` kernels are pure data-parallel compute — exactly what GPUs were born to do. Spectral graph analysis on a 10,000-node graph that takes seconds on CPU completes in milliseconds on GPU.

## How to Use

```bash
# Install OpenCL headers and ICD loader
sudo apt install opencl-headers ocl-icd-opencl-dev

# Build
make

# Run tests
make test
```

Expected output: a 5-node chord graph analysis showing eigenvalues, spectral gap, Cheeger constant, conservation ratios, and anomaly detection.

## Architecture

- **OpenCL kernels** (`kernels/*.cl`): `laplacian`, `power_iteration`, `conservation`, `fiedler`, `anomaly`
- **C host code** (`src/`): Platform/device detection, kernel compilation, data marshalling
- Zero dependencies beyond OpenCL ICD + standard C math/string

### Key Kernels

| Kernel | Purpose |
|--------|---------|
| `laplacian` | Build L = D - W from adjacency matrix |
| `power_iteration` | Find dominant eigenvector via iterative GPU multiply |
| `conservation` | Compute α(G,a) = (a^T L a) / (λ_max ‖a‖²) |
| `fiedler` | Compute the Fiedler vector (eigenvector of λ₂) |
| `anomaly` | Detect conservation anomalies via residual analysis |

## Connection to the Conservation Spectral Framework

This is the **cross-vendor GPU** implementation. It implements the same conservation ratio α(G,a) pipeline as the Vulkan, PTX, and WebGPU versions, but prioritizes portability over peak performance. Where Vulkan gives you the lowest-level cross-platform control and PTX lets you hand-write GPU assembly, OpenCL gives you *write once, run on any GPU*.

## Related Repos

- [conservation-spectral-vulkan](https://github.com/SuperInstance/conservation-spectral-vulkan) — Lowest-level cross-platform GPU
- [conservation-spectral-ptx](https://github.com/SuperInstance/conservation-spectral-ptx) — Hand-written NVIDIA GPU assembly
- [conservation-spectral-webgpu](https://github.com/SuperInstance/conservation-spectral-webgpu) — Browser-based spectral analysis
- [conservation-spectral-v2](https://github.com/SuperInstance/conservation-spectral-v2) — Reference Python implementation
- [conservation-spectral-asm](https://github.com/SuperInstance/conservation-spectral-asm) — CPU AVX2 assembly

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
