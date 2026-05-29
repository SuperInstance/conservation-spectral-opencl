/*
 * anomaly.cl — Spectral anomaly detection on GPU
 *
 * Score each vertex based on deviation from expected conservation:
 *   score[i] = |fiedler[i]| * attribute_deviation[i]
 *
 * where attribute_deviation is how far the vertex's attribute is from
 * the mean of its community (same Fiedler sign group).
 *
 * High score → the vertex's attribute is misaligned with its spectral community.
 */

/* Compute anomaly scores for all vertices.
 * Each work-item handles one vertex. */
__kernel void anomaly_scores(
    __global const double *fiedler,         /* [n] */
    __global const double *attr,            /* [n] */
    __global const double *eigenvectors,    /* [n*n] column-major */
    __global double       *scores,          /* [n] output anomaly scores */
    const uint             n)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    /* Compute community means (single-pass approximation) */
    double pos_sum = 0.0, neg_sum = 0.0;
    uint pos_count = 0, neg_count = 0;

    for (uint j = 0; j < n; j++) {
        if (fiedler[j] >= 0.0) {
            pos_sum += attr[j];
            pos_count++;
        } else {
            neg_sum += attr[j];
            neg_count++;
        }
    }

    double pos_mean = (pos_count > 0) ? pos_sum / (double)pos_count : 0.0;
    double neg_mean = (neg_count > 0) ? neg_sum / (double)neg_count : 0.0;

    /* Expected attribute = mean of same-sign community */
    double expected = (fiedler[i] >= 0.0) ? pos_mean : neg_mean;
    double deviation = fabs(attr[i] - expected);

    /* Weight by Fiedler magnitude (vertices near the cut are more anomalous
     * if they also deviate in attribute) */
    double fiedler_weight = fabs(fiedler[i]);

    /* Also consider projection onto 3rd eigenvector (captures finer structure) */
    double ev3_proj = 0.0;
    if (n >= 3) {
        ev3_proj = fabs(eigenvectors[(size_t)2 * n + i]);
    }

    scores[i] = deviation * (1.0 + fiedler_weight) * (1.0 + ev3_proj);
}

/* Find vertices with anomaly score above threshold.
 * Returns indices and scores. Single work-item. */
__kernel void anomaly_threshold(
    __global const double *scores,          /* [n] */
    __global uint         *anomaly_ids,     /* [n] output vertex indices */
    __global double       *anomaly_scores,  /* [n] output scores */
    __global uint         *anomaly_count,   /* [1] output count */
    const uint             n,
    const double           threshold)
{
    if (get_global_id(0) != 0) return;

    uint count = 0;
    for (uint i = 0; i < n; i++) {
        if (scores[i] > threshold) {
            anomaly_ids[count] = i;
            anomaly_scores[count] = scores[i];
            count++;
        }
    }
    *anomaly_count = count;
}
