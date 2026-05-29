/*
 * conservation.cl — Conservation ratio computation on GPU
 *
 * For eigenvector φ_k and attribute vector a:
 *   projection[i] = a[i] * φ_k[i]
 *   gradient[i]   = projection[i+1] - projection[i]
 *   ratio         = var(gradient)
 *
 * Single work-item kernel (n is small for spectral graphs).
 */

__kernel void conservation_ratio(
    __global const double *eigenvector, /* [n] one eigenvector */
    __global const double *attr,        /* [n] attribute values */
    __global double       *ratio_out,   /* [1] output variance */
    const uint             n)
{
    if (get_global_id(0) != 0) return;

    if (n < 2) {
        *ratio_out = 0.0;
        return;
    }

    /* Compute projections */
    double proj[1024];  /* stack alloc for small graphs */
    __global const double *ev = eigenvector;

    for (uint i = 0; i < n && i < 1024; i++)
        proj[i] = attr[i] * ev[i];

    /* Compute gradient */
    double grad[1024];
    uint gn = n - 1;
    for (uint i = 0; i < gn && i < 1024; i++)
        grad[i] = proj[i + 1] - proj[i];

    /* Variance of gradient */
    double mean = 0.0;
    for (uint i = 0; i < gn; i++)
        mean += grad[i];
    mean /= (double)gn;

    double var = 0.0;
    for (uint i = 0; i < gn; i++) {
        double d = grad[i] - mean;
        var += d * d;
    }
    var /= (double)gn;

    *ratio_out = var;
}

/* Batch: compute conservation ratios for all eigenvectors at once.
 * Each work-item handles one eigenvector. */
__kernel void conservation_ratio_batch(
    __global const double *eigenvectors, /* [n*n] column-major */
    __global const double *attr,         /* [n] */
    __global double       *ratios,       /* [n] output */
    const uint             n)
{
    uint k = get_global_id(0);
    if (k >= n) return;

    if (n < 2) {
        ratios[k] = 0.0;
        return;
    }

    /* Pointer to eigenvector k (column-major: starts at k*n) */
    __global const double *ev = &eigenvectors[(size_t)k * n];

    /* Projection */
    double mean = 0.0;
    double prev = attr[0] * ev[0];

    /* Gradient mean */
    double gmean = 0.0;
    for (uint i = 1; i < n; i++) {
        double cur = attr[i] * ev[i];
        double g = cur - prev;
        gmean += g;
        prev = cur;
    }
    gmean /= (double)(n - 1);

    /* Gradient variance */
    prev = attr[0] * ev[0];
    double var = 0.0;
    for (uint i = 1; i < n; i++) {
        double cur = attr[i] * ev[i];
        double g = cur - prev;
        double d = g - gmean;
        var += d * d;
        prev = cur;
    }
    var /= (double)(n - 1);

    ratios[k] = var;
}
