# Conservation Spectral SDK — OpenCL

Cross-vendor GPU compute for spectral graph analysis. Laplacian construction, power iteration eigen decomposition, conservation ratios, Fiedler vector, and anomaly detection — all accelerated on GPU via OpenCL.

## Build

```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
make
```

## Run Tests

```bash
make test
```

Expected output: a 5-node chord graph analysis showing eigenvalues, spectral gap, Cheeger constant, conservation ratios, and anomaly detection.

## Architecture

- **OpenCL kernels** (`kernels/*.cl`): `laplacian`, `power_iteration`, `conservation`, `fiedler`, `anomaly`
- **C host code** (`src/`): Platform/device detection, kernel compilation, data marshalling
- Zero dependencies beyond OpenCL ICD + standard C math/string

## License

MIT
