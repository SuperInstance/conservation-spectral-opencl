/*
 * fiedler.cl — Fiedler vector extraction and Cheeger constant on GPU
 *
 * The Fiedler vector is the eigenvector corresponding to the second-smallest
 * eigenvalue (λ₂) of the graph Laplacian. It partitions the graph into
 * two communities based on sign.
 *
 * Cheeger constant h(G) ≈ |∂S| / min(vol(S), vol(V\S))
 * where S = {i : fiedler[i] < 0}.
 */

/* Compute Cheeger constant from Fiedler vector and Laplacian.
 * Single work-item kernel. */
__kernel void cheeger_constant(
    __global const double *lap_values,  /* [n*n] Laplacian */
    __global const double *fiedler,     /* [n] Fiedler vector */
    __global double       *h_out,       /* [1] output Cheeger constant */
    const uint             n)
{
    if (get_global_id(0) != 0) return;

    /* Partition: S = {i : fiedler[i] < 0} */
    double cut = 0.0;      /* |∂S| = sum of edge weights crossing the cut */
    double vol_s = 0.0;    /* vol(S) = sum of degrees in S */

    for (uint i = 0; i < n; i++) {
        bool in_s = (fiedler[i] < 0.0);
        if (!in_s) continue;

        for (uint j = 0; j < n; j++) {
            if (i == j) continue;
            double w = -lap_values[(size_t)i * n + j];  /* off-diagonal = -weight */
            vol_s += w;  /* degree contribution */
            if (fiedler[j] >= 0.0) {
                /* Edge crosses from S to complement */
                cut += w;
            }
        }
    }

    /* Total volume = sum of diagonal (= sum of degrees) */
    double total_vol = 0.0;
    for (uint i = 0; i < n; i++)
        total_vol += lap_values[(size_t)i * n + i];

    double vol_comp = total_vol - vol_s;
    double min_vol = (vol_s < vol_comp) ? vol_s : vol_comp;

    if (min_vol < 1e-15) {
        *h_out = 0.0;
    } else {
        *h_out = cut / min_vol;
    }
}

/* Extract Fiedler vector (eigenvector for 2nd smallest eigenvalue).
 * The eigenvectors are stored column-major in eigenvectors matrix.
 * Column 1 (index 1) is the Fiedler vector (after sorting by eigenvalue). */
__kernel void extract_fiedler(
    __global const double *eigenvectors, /* [n*n] column-major */
    __global double       *fiedler,     /* [n] output */
    const uint             n)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    /* Column 1 in column-major = index [1*n + i] */
    fiedler[i] = eigenvectors[(size_t)1 * n + i];
}
