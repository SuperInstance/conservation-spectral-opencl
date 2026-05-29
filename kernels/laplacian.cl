/*
 * laplacian.cl — Build graph Laplacian on GPU
 */

/* Build W matrix from edge list (serial — works for any OpenCL version). */
__kernel void build_adjacency_serial(
    __global const uint    *from,
    __global const uint    *to,
    __global const double  *weight,
    __global double        *W,           /* [n*n] initialized to 0 */
    const uint              n,
    const uint              n_edges)
{
    for (uint e = 0; e < n_edges; e++) {
        uint i = from[e];
        uint j = to[e];
        double w = weight[e];
        W[(size_t)i * n + j] += w;
        W[(size_t)j * n + i] += w;
    }
}

/* Compute unnormalized Laplacian L = D - W. */
__kernel void laplacian_unnormalized(
    __global const double *W,
    __global double       *L,
    const uint             n)
{
    size_t gid = get_global_id(0);
    if (gid >= (size_t)n * n) return;

    size_t i = gid / n;
    size_t j = gid % n;

    if (i == j) {
        double deg = 0.0;
        for (uint k = 0; k < n; k++)
            deg += W[i * n + k];
        L[gid] = deg;
    } else {
        L[gid] = -W[gid];
    }
}

/* Compute symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2}. */
__kernel void laplacian_normalized(
    __global const double *W,
    __global double       *L,
    const uint             n)
{
    size_t gid = get_global_id(0);
    if (gid >= (size_t)n * n) return;

    size_t i = gid / n;
    size_t j = gid % n;

    double deg_i = 0.0;
    for (uint k = 0; k < n; k++)
        deg_i += W[i * n + k];

    if (i == j) {
        double di_sqrt = (deg_i > 0.0) ? 1.0 / sqrt(deg_i) : 0.0;
        double val = 0.0;
        for (uint k = 0; k < n; k++)
            val += di_sqrt * W[i * n + k] * di_sqrt;
        L[gid] = 1.0 - val;
    } else {
        double deg_j = 0.0;
        for (uint k = 0; k < n; k++)
            deg_j += W[j * n + k];

        double di_sqrt = (deg_i > 0.0) ? 1.0 / sqrt(deg_i) : 0.0;
        double dj_sqrt = (deg_j > 0.0) ? 1.0 / sqrt(deg_j) : 0.0;
        L[gid] = -di_sqrt * W[gid] * dj_sqrt;
    }
}
