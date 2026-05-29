/*
 * power_iteration.cl — Power iteration eigen decomposition on GPU
 *
 * Computes k largest eigenvalues/eigenvectors of a shifted matrix
 * M = shift*I - L via power iteration with Hotelling deflation.
 *
 * The eigenvalues of L are recovered as: λ_L = shift - λ_M
 *
 * Kernels:
 *   1. matvec:       y = A*x  (dense n×n)
 *   2. normalize:    x = x / ||x||
 *   3. rayleigh:     λ = x^T A x
 *   4. deflate:      A = A - λ * x*x^T
 *   5. shift_matrix: M = shift*I - L
 *   6. init_vector:  x = seed values
 */

/* Dense matrix-vector multiply: y = A * x
 * Each work-item computes one element of y. */
__kernel void matvec(
    __global const double *A,       /* [n*n] row-major */
    __global const double *x,       /* [n] */
    __global double       *y,       /* [n] output */
    const uint             n)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    double sum = 0.0;
    for (uint j = 0; j < n; j++) {
        sum += A[(size_t)i * n + j] * x[j];
    }
    y[i] = sum;
}

/* Normalize a vector in-place: x = x / ||x||
 * Single work-item reduction (sufficient for n <= ~500). */
__kernel void normalize(
    __global double *x,             /* [n] in-place */
    __global double *out_norm,      /* [1] output norm value */
    const uint       n)
{
    if (get_global_id(0) != 0) return;

    double norm_sq = 0.0;
    for (uint i = 0; i < n; i++)
        norm_sq += x[i] * x[i];

    double norm = sqrt(norm_sq);
    *out_norm = norm;

    if (norm > 1e-30) {
        double inv = 1.0 / norm;
        for (uint i = 0; i < n; i++)
            x[i] *= inv;
    }
}

/* Rayleigh quotient: returns λ = x^T * A * x
 * Single work-item for simplicity. */
__kernel void rayleigh(
    __global const double *A,       /* [n*n] */
    __global const double *x,       /* [n] */
    __global double       *lambda,  /* [1] output */
    const uint             n)
{
    if (get_global_id(0) != 0) return;

    /* Compute y = A*x first */
    double rq = 0.0;
    for (uint i = 0; i < n; i++) {
        double yi = 0.0;
        for (uint j = 0; j < n; j++)
            yi += A[(size_t)i * n + j] * x[j];
        rq += x[i] * yi;
    }
    *lambda = rq;
}

/* Deflation: R = R - lambda * v*v^T
 * Each work-item updates one element. */
__kernel void deflate(
    __global double       *R,       /* [n*n] in-place residual matrix */
    __global const double *v,       /* [n] eigenvector */
    const double           lambda,
    const uint             n)
{
    size_t gid = get_global_id(0);
    if (gid >= (size_t)n * n) return;

    size_t i = gid / n;
    size_t j = gid % n;

    R[gid] -= lambda * v[i] * v[j];
}

/* Build shifted matrix: M = shift*I - L
 * Each work-item computes one element. */
__kernel void shift_matrix(
    __global const double *L,       /* [n*n] Laplacian */
    __global double       *M,       /* [n*n] output */
    const double           shift,
    const uint             n)
{
    size_t gid = get_global_id(0);
    if (gid >= (size_t)n * n) return;

    size_t i = gid / n;
    M[gid] = -L[gid];
    if (i == (gid % n))  /* diagonal */
        M[gid] += shift;
}

/* Initialize vector with varied seed values.
 * seed_base prevents all eigenvectors from starting the same. */
__kernel void init_vector(
    __global double *x,             /* [n] output */
    const uint       n,
    const uint       seed_base)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    x[i] = 1.0 / (double)(i + 1 + seed_base);
}

/* Full power iteration for one eigenvector.
 * Runs entirely on GPU — no host round-trips per iteration.
 * Single work-item kernel that iterates max_iter times. */
__kernel void power_iteration_one(
    __global const double *R,       /* [n*n] residual matrix (read-only here) */
    __global double       *v,       /* [n] eigenvector (in/out) */
    __global double       *w,       /* [n] workspace */
    __global double       *lambda_out, /* [1] output eigenvalue of R */
    const uint             n,
    const uint             max_iter,
    const uint             seed_base)
{
    if (get_global_id(0) != 0) return;

    /* Init vector */
    for (uint i = 0; i < n; i++)
        v[i] = 1.0 / (double)(i + 1 + seed_base);

    double lambda = 0.0;
    double tol = 1e-12;

    for (uint iter = 0; iter < max_iter; iter++) {
        /* w = R * v */
        for (uint i = 0; i < n; i++) {
            double sum = 0.0;
            for (uint j = 0; j < n; j++)
                sum += R[(size_t)i * n + j] * v[j];
            w[i] = sum;
        }

        /* Normalize v = w / ||w|| */
        double norm_sq = 0.0;
        for (uint i = 0; i < n; i++)
            norm_sq += w[i] * w[i];
        double norm = sqrt(norm_sq);
        if (norm < 1e-30) break;
        for (uint i = 0; i < n; i++)
            v[i] = w[i] / norm;

        /* Rayleigh quotient: λ = v^T R v */
        double rq = 0.0;
        for (uint i = 0; i < n; i++) {
            double yi = 0.0;
            for (uint j = 0; j < n; j++)
                yi += R[(size_t)i * n + j] * v[j];
            rq += v[i] * yi;
        }

        if (fabs(rq - lambda) < tol) {
            lambda = rq;
            break;
        }
        lambda = rq;
    }

    *lambda_out = lambda;
}
