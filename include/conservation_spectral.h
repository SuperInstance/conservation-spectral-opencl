/*
 * conservation_spectral.h — Conservation Spectral SDK for C
 *
 * Header-only library (STB-style). In exactly ONE .c file, do:
 *
 *     #define CS_IMPLEMENTATION
 *     #include "conservation_spectral.h"
 *
 * All other files just #include "conservation_spectral.h" without the define.
 *
 * Zero dependencies beyond math.h and string.h.
 *
 * Version: 0.1.0
 */

#ifndef CONSERVATION_SPECTRAL_H
#define CONSERVATION_SPECTRAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Version
 * ============================================================ */

#define CS_VERSION_MAJOR 0
#define CS_VERSION_MINOR 1
#define CS_VERSION_PATCH 0

/* ============================================================
 * Error codes
 * ============================================================ */

typedef enum {
    CS_OK = 0,
    CS_ERR_NULL_POINTER = 1,
    CS_ERR_OUT_OF_BOUNDS = 2,
    CS_ERR_INVALID_STATE = 3,
    CS_ERR_ALLOC_FAILED = 4,
    CS_ERR_DIMENSION_MISMATCH = 5,
    CS_ERR_SINGULAR = 6,
    CS_ERR_NO_CONVERGENCE = 7,
} CsError;

/* ============================================================
 * Graph
 * ============================================================ */

typedef struct {
    uint64_t id;
    double   attribute;   /* primary scalar attribute per vertex */
} cs_vertex;

typedef struct {
    size_t   from;
    size_t   to;
    double   weight;
} cs_edge;

typedef struct {
    size_t      n_vertices;
    size_t      n_edges;
    size_t      cap_edges;
    cs_vertex  *vertices;
    cs_edge    *edges;
    bool        directed;
} cs_graph;

cs_graph *cs_graph_create(size_t n_vertices);
void      cs_graph_free(cs_graph *g);
int       cs_graph_add_vertex(cs_graph *g, uint64_t id, double attr);
int       cs_graph_add_edge(cs_graph *g, size_t from, size_t to, double weight);

/* ============================================================
 * Laplacian (dense, row-major)
 * ============================================================ */

typedef struct {
    double *values;       /* row-major n×n matrix */
    size_t  n;
    bool    normalized;
} cs_laplacian;

void        cs_laplacian_free(cs_laplacian *l);
cs_laplacian cs_build_laplacian(const cs_graph *g, bool normalized);

/* ============================================================
 * Eigen decomposition
 * ============================================================ */

typedef struct {
    double *eigenvalues;   /* length n, sorted ascending */
    double *eigenvectors;  /* n×n column-major: col i = eigenvector i */
    size_t  n;
} cs_eigen;

void    cs_eigen_free(cs_eigen *e);
cs_eigen cs_eigendecompose(const cs_laplacian *l, size_t k);

/* ============================================================
 * Conservation analysis
 * ============================================================ */

typedef struct {
    size_t eigenvector_index;
    double eigenvalue;
    double ratio;
} cs_ratio;

typedef struct {
    double spectral_gap;
    double cheeger_constant;
    size_t n_ratios;
    cs_ratio *ratios;
} cs_report;

void      cs_report_free(cs_report *r);
cs_report cs_analyze(const cs_graph *g, const char *attribute_name);

/* Compute conservation ratio for one eigenvector against an attribute array */
double cs_conservation_ratio(const cs_eigen *eigen, const double *attr,
                             size_t attr_len, size_t eigenvector_index);

/* Spectral gap = largest gap between consecutive eigenvalues */
double cs_spectral_gap(const cs_eigen *eigen);

/* Cheeger constant approximation from Fiedler vector */
double cs_cheeger_constant(const cs_laplacian *lap, const double *fiedler);

/* ============================================================
 * Sliding-window tracker
 * ============================================================ */

typedef struct {
    size_t  window_size;
    size_t  count;
    double *history;
    double  baseline_mean;
    double  baseline_std;
    bool    baseline_set;
} cs_tracker;

cs_tracker *cs_tracker_create(size_t window_size);
void        cs_tracker_free(cs_tracker *t);

/* Feed a single observation value. Returns 0=nominal, 1=warning, 2=critical */
int         cs_tracker_feed(cs_tracker *t, double observation);

/* Check current state without feeding */
int         cs_tracker_check(const cs_tracker *t);

/* ============================================================
 * Spectral fingerprint
 * ============================================================ */

/* Compute hex fingerprint string from eigenvalues. Caller must free(). */
char  *cs_fingerprint_compute(const double *eigenvalues, size_t n);

/* Compare two fingerprints. Returns similarity in [0,1]. */
double cs_fingerprint_compare(const char *fp1, const char *fp2);

/* ============================================================
 * Anomaly
 * ============================================================ */

typedef struct {
    uint64_t vertex_id;
    size_t   eigenvector_index;
    double   score;
    char     suggested_fix[256];
} cs_anomaly;

#ifdef __cplusplus
}
#endif

#endif /* CONSERVATION_SPECTRAL_H */


/* ============================================================
 * IMPLEMENTATION
 * ============================================================ */

#ifdef CS_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>

/* ---- helpers ---- */

static double *cs_alloc_matrix(size_t n) {
    double *m = (double *)calloc(n * n, sizeof(double));
    return m;
}

/* ---- Graph ---- */

cs_graph *cs_graph_create(size_t n_vertices) {
    cs_graph *g = (cs_graph *)calloc(1, sizeof(cs_graph));
    if (!g) return NULL;
    g->n_vertices = n_vertices;
    g->vertices = (cs_vertex *)calloc(n_vertices, sizeof(cs_vertex));
    g->cap_edges = n_vertices * 2;
    g->edges = (cs_edge *)calloc(g->cap_edges, sizeof(cs_edge));
    g->n_edges = 0;
    g->directed = false;
    return g;
}

void cs_graph_free(cs_graph *g) {
    if (!g) return;
    free(g->vertices);
    free(g->edges);
    free(g);
}

int cs_graph_add_vertex(cs_graph *g, uint64_t id, double attr) {
    if (!g || id >= g->n_vertices) return -1;
    g->vertices[id].id = id;
    g->vertices[id].attribute = attr;
    return 0;
}

int cs_graph_add_edge(cs_graph *g, size_t from, size_t to, double weight) {
    if (!g || from >= g->n_vertices || to >= g->n_vertices) return -1;
    if (g->n_edges >= g->cap_edges) {
        size_t new_cap = g->cap_edges * 2;
        cs_edge *tmp = (cs_edge *)realloc(g->edges, new_cap * sizeof(cs_edge));
        if (!tmp) return -2;
        g->edges = tmp;
        g->cap_edges = new_cap;
    }
    g->edges[g->n_edges].from = from;
    g->edges[g->n_edges].to = to;
    g->edges[g->n_edges].weight = weight;
    g->n_edges++;
    return 0;
}

/* ---- Laplacian ---- */

void cs_laplacian_free(cs_laplacian *l) {
    if (!l) return;
    free(l->values);
    l->values = NULL;
}

cs_laplacian cs_build_laplacian(const cs_graph *g, bool normalized) {
    cs_laplacian lap;
    memset(&lap, 0, sizeof(lap));
    lap.n = g->n_vertices;
    lap.normalized = normalized;
    lap.values = cs_alloc_matrix(lap.n);
    if (!lap.values) return lap;

    /* Build adjacency weight matrix */
    double *W = cs_alloc_matrix(lap.n);
    if (!W) return lap;

    for (size_t e = 0; e < g->n_edges; e++) {
        size_t i = g->edges[e].from;
        size_t j = g->edges[e].to;
        double w = g->edges[e].weight;
        W[i * lap.n + j] += w;
        W[j * lap.n + i] += w;  /* undirected */
    }

    /* Degree vector */
    double *deg = (double *)calloc(lap.n, sizeof(double));
    for (size_t i = 0; i < lap.n; i++)
        for (size_t j = 0; j < lap.n; j++)
            deg[i] += W[i * lap.n + j];

    if (!normalized) {
        /* L = D - W */
        for (size_t i = 0; i < lap.n; i++) {
            for (size_t j = 0; j < lap.n; j++) {
                if (i == j)
                    lap.values[i * lap.n + j] = deg[i] - W[i * lap.n + j];
                else
                    lap.values[i * lap.n + j] = -W[i * lap.n + j];
            }
        }
    } else {
        /* Symmetric normalized: L = I - D^{-1/2} W D^{-1/2} */
        for (size_t i = 0; i < lap.n; i++) {
            for (size_t j = 0; j < lap.n; j++) {
                double di_sqrt = deg[i] > 0.0 ? 1.0 / sqrt(deg[i]) : 0.0;
                double dj_sqrt = deg[j] > 0.0 ? 1.0 / sqrt(deg[j]) : 0.0;
                double val = di_sqrt * W[i * lap.n + j] * dj_sqrt;
                if (i == j)
                    lap.values[i * lap.n + j] = 1.0 - val;
                else
                    lap.values[i * lap.n + j] = -val;
            }
        }
    }

    free(W);
    free(deg);
    return lap;
}

/* ---- Dense matrix-vector multiply: y = A*x ---- */
static void cs_matvec(const double *A, const double *x, double *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

/* ---- Power iteration with deflation for k smallest eigenvalues ----
 * For the Laplacian, the smallest eigenvalue is 0 (trivial).
 * We compute eigenvalues by targeting the largest eigenvalues of
 * (shifted) matrix: M = shift*I - L, where shift = max diagonal of L.
 * Then eigenvalues of L = shift - eigenvalues of M.
 */

cs_eigen cs_eigendecompose(const cs_laplacian *l, size_t k) {
    cs_eigen eig;
    memset(&eig, 0, sizeof(eig));
    size_t n = l->n;
    eig.n = n;
    if (k == 0 || k > n) k = n;

    eig.eigenvalues = (double *)calloc(n, sizeof(double));
    eig.eigenvectors = (double *)calloc(n * n, sizeof(double));

    if (!eig.eigenvalues || !eig.eigenvectors) return eig;

    /* Find shift = max diagonal element of L (for positive semi-definite, this
       upper-bounds the max eigenvalue) */
    double shift = 0.0;
    for (size_t i = 0; i < n; i++)
        if (l->values[i * n + i] > shift) shift = l->values[i * n + i];

    /* Build M = shift*I - L */
    double *M = cs_alloc_matrix(n);
    if (!M) return eig;
    for (size_t i = 0; i < n * n; i++)
        M[i] = -l->values[i];
    for (size_t i = 0; i < n; i++)
        M[i * n + i] += shift;

    /* Residual matrix for deflation */
    double *R = cs_alloc_matrix(n);
    if (!R) { free(M); return eig; }
    memcpy(R, M, n * n * sizeof(double));

    double *v = (double *)calloc(n, sizeof(double));
    double *w = (double *)calloc(n, sizeof(double));
    double *q = (double *)calloc(n, sizeof(double));

    if (!v || !w || !q) {
        free(M); free(R); free(v); free(w); free(q);
        return eig;
    }

    /* Find k largest eigenvalues of M via power iteration + deflation */
    for (size_t ev = 0; ev < k; ev++) {
        /* Initial vector */
        for (size_t i = 0; i < n; i++)
            v[i] = 1.0 / (double)(i + 1 + ev);  /* varied seeds */

        /* Power iteration */
        size_t max_iter = 2000;
        double tol = 1e-12;
        double lambda = 0.0;

        for (size_t iter = 0; iter < max_iter; iter++) {
            cs_matvec(R, v, w, n);

            /* Normalize */
            double norm = 0.0;
            for (size_t i = 0; i < n; i++) norm += w[i] * w[i];
            norm = sqrt(norm);
            if (norm < 1e-30) break;

            for (size_t i = 0; i < n; i++) v[i] = w[i] / norm;

            /* Rayleigh quotient */
            cs_matvec(R, v, w, n);
            double rq = 0.0;
            for (size_t i = 0; i < n; i++) rq += v[i] * w[i];

            if (fabs(rq - lambda) < tol) { lambda = rq; break; }
            lambda = rq;
        }

        /* Store: eigenvalue of L = shift - lambda_M */
        eig.eigenvalues[ev] = shift - lambda;

        /* Store eigenvector (column ev in column-major) */
        for (size_t i = 0; i < n; i++)
            eig.eigenvectors[ev * n + i] = v[i];

        /* Deflate: R = R - lambda * v*v^T */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                R[i * n + j] -= lambda * v[i] * v[j];
            }
        }
    }

    free(M); free(R); free(v); free(w); free(q);

    /* Sort eigenvalues ascending (and corresponding eigenvectors) */
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (eig.eigenvalues[j] < eig.eigenvalues[i]) {
                double tmp = eig.eigenvalues[i];
                eig.eigenvalues[i] = eig.eigenvalues[j];
                eig.eigenvalues[j] = tmp;
                /* swap columns i and j */
                for (size_t r = 0; r < n; r++) {
                    double tv = eig.eigenvectors[i * n + r];
                    eig.eigenvectors[i * n + r] = eig.eigenvectors[j * n + r];
                    eig.eigenvectors[j * n + r] = tv;
                }
            }
        }
    }

    return eig;
}

void cs_eigen_free(cs_eigen *e) {
    if (!e) return;
    free(e->eigenvalues);
    free(e->eigenvectors);
    e->eigenvalues = NULL;
    e->eigenvectors = NULL;
}

/* ---- Conservation ratio ---- */

double cs_conservation_ratio(const cs_eigen *eigen, const double *attr,
                             size_t attr_len, size_t eigenvector_index) {
    if (!eigen || !attr || eigen->n != attr_len) return -1.0;
    size_t n = eigen->n;

    /* Project attribute onto eigenvector */
    /* Eigenvector is column eigenvector_index in column-major: stored at [eigenvector_index * n + i] */
    double *projection = (double *)calloc(n, sizeof(double));
    if (!projection) return -1.0;

    for (size_t i = 0; i < n; i++)
        projection[i] = attr[i] * eigen->eigenvectors[eigenvector_index * n + i];

    /* Gradient: diff of consecutive projected values */
    if (n < 2) { free(projection); return 0.0; }

    double *gradient = (double *)calloc(n - 1, sizeof(double));
    if (!gradient) { free(projection); return -1.0; }

    for (size_t i = 0; i < n - 1; i++)
        gradient[i] = projection[i + 1] - projection[i];

    /* Variance of gradient */
    double mean = 0.0;
    for (size_t i = 0; i < n - 1; i++) mean += gradient[i];
    mean /= (double)(n - 1);

    double var = 0.0;
    for (size_t i = 0; i < n - 1; i++) {
        double d = gradient[i] - mean;
        var += d * d;
    }
    var /= (double)(n - 1);

    free(projection);
    free(gradient);
    return var;
}

/* ---- Spectral gap ---- */

double cs_spectral_gap(const cs_eigen *eigen) {
    if (!eigen || eigen->n < 2) return 0.0;
    double max_gap = 0.0;
    for (size_t i = 0; i < eigen->n - 1; i++) {
        double gap = eigen->eigenvalues[i + 1] - eigen->eigenvalues[i];
        if (gap > max_gap) max_gap = gap;
    }
    return max_gap;
}

/* ---- Cheeger constant (approximation from Fiedler vector) ---- */

double cs_cheeger_constant(const cs_laplacian *lap, const double *fiedler) {
    if (!lap || !fiedler) return 0.0;
    size_t n = lap->n;

    /* Partition vertices into S and complement based on Fiedler vector sign */
    size_t *in_s = (size_t *)calloc(n, sizeof(size_t));
    if (!in_s) return 0.0;

    for (size_t i = 0; i < n; i++)
        in_s[i] = (fiedler[i] < 0.0) ? 1 : 0;

    /* Count edges crossing the cut and edges within S */
    double cut = 0.0;
    double vol_s = 0.0;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double w = -lap->values[i * n + j];
            if (i != j) {
                /* unused: double deg_ij = (i == j) ? 0.0 : w; */
                if (in_s[i]) {
                    vol_s += (i == j) ? lap->values[i * n + i] : -lap->values[i * n + j];
                    if (i == j) vol_s += lap->values[i * n + i];
                }
                if (in_s[i] && !in_s[j]) cut += w;
            }
        }
    }

    /* Cheeger = cut / min(vol_S, vol_complement) */
    double total_vol = 0.0;
    for (size_t i = 0; i < n; i++)
        total_vol += lap->values[i * n + i];  /* diagonal = degree */

    double vol_comp = total_vol - vol_s;
    double min_vol = vol_s < vol_comp ? vol_s : vol_comp;

    free(in_s);

    if (min_vol < 1e-15) return 0.0;
    return cut / min_vol;
}

/* ---- Full analysis ---- */

cs_report cs_analyze(const cs_graph *g, const char *attribute_name) {
    (void)attribute_name;
    cs_report rpt;
    memset(&rpt, 0, sizeof(rpt));

    cs_laplacian lap = cs_build_laplacian(g, false);
    if (!lap.values) return rpt;

    cs_eigen eig = cs_eigendecompose(&lap, 0);
    if (!eig.eigenvalues) { cs_laplacian_free(&lap); return rpt; }

    /* Collect vertex attributes */
    double *attr = (double *)calloc(g->n_vertices, sizeof(double));
    if (!attr) { cs_laplacian_free(&lap); cs_eigen_free(&eig); return rpt; }

    for (size_t i = 0; i < g->n_vertices; i++)
        attr[i] = g->vertices[i].attribute;

    /* Compute ratios */
    rpt.n_ratios = eig.n;
    rpt.ratios = (cs_ratio *)calloc(rpt.n_ratios, sizeof(cs_ratio));
    for (size_t k = 0; k < rpt.n_ratios; k++) {
        rpt.ratios[k].eigenvector_index = k;
        rpt.ratios[k].eigenvalue = eig.eigenvalues[k];
        rpt.ratios[k].ratio = cs_conservation_ratio(&eig, attr, g->n_vertices, k);
    }

    rpt.spectral_gap = cs_spectral_gap(&eig);

    /* Fiedler vector = eigenvector for 2nd smallest eigenvalue (index 1) */
    if (eig.n >= 2) {
        double *fiedler = &eig.eigenvectors[1 * eig.n];
        rpt.cheeger_constant = cs_cheeger_constant(&lap, fiedler);
    }

    free(attr);
    cs_laplacian_free(&lap);
    cs_eigen_free(&eig);
    return rpt;
}

void cs_report_free(cs_report *r) {
    if (!r) return;
    free(r->ratios);
    r->ratios = NULL;
}

/* ---- Tracker ---- */

cs_tracker *cs_tracker_create(size_t window_size) {
    cs_tracker *t = (cs_tracker *)calloc(1, sizeof(cs_tracker));
    if (!t) return NULL;
    t->window_size = window_size;
    t->history = (double *)calloc(window_size, sizeof(double));
    t->count = 0;
    t->baseline_mean = 0.0;
    t->baseline_std = 0.0;
    t->baseline_set = false;
    return t;
}

void cs_tracker_free(cs_tracker *t) {
    if (!t) return;
    free(t->history);
    free(t);
}

int cs_tracker_feed(cs_tracker *t, double observation) {
    if (!t) return 0;

    /* Sliding window */
    if (t->count < t->window_size) {
        t->history[t->count] = observation;
        t->count++;
    } else {
        /* Shift left */
        memmove(t->history, t->history + 1, (t->window_size - 1) * sizeof(double));
        t->history[t->window_size - 1] = observation;
    }

    /* Establish baseline after filling window once */
    if (t->count == t->window_size && !t->baseline_set) {
        double sum = 0.0;
        for (size_t i = 0; i < t->window_size; i++) sum += t->history[i];
        t->baseline_mean = sum / (double)t->window_size;

        double var = 0.0;
        for (size_t i = 0; i < t->window_size; i++) {
            double d = t->history[i] - t->baseline_mean;
            var += d * d;
        }
        t->baseline_std = sqrt(var / (double)t->window_size);
        t->baseline_set = true;
        return 0; /* just established baseline */
    }

    return cs_tracker_check(t);
}

int cs_tracker_check(const cs_tracker *t) {
    if (!t || !t->baseline_set || t->count == 0) return 0;

    double latest = t->history[t->count - 1];
    if (t->baseline_std < 1e-15) return 0;

    double zscore = fabs(latest - t->baseline_mean) / t->baseline_std;

    if (zscore > 3.0) return 2;  /* critical */
    if (zscore > 2.0) return 1;  /* warning */
    return 0;                    /* nominal */
}

/* ---- Spectral fingerprint ---- */

char *cs_fingerprint_compute(const double *eigenvalues, size_t n) {
    if (!eigenvalues || n == 0) return NULL;

    /* Hash eigenvalues into hex: XOR-fold quantized values */
    /* Use a simple hashing: sum of quantized eigenvalue bits */
    size_t hex_len = n * 16 + 1;
    char *hex = (char *)calloc(hex_len, sizeof(char));
    if (!hex) return NULL;

    size_t pos = 0;
    for (size_t i = 0; i < n && pos < hex_len - 17; i++) {
        /* Quantize to uint64 via bit-cast of double, mix bits */
        uint64_t bits;
        memcpy(&bits, &eigenvalues[i], sizeof(double));
        bits ^= (bits >> 33);
        bits *= 0xff51afd7ed558ccdULL;
        bits ^= (bits >> 33);
        bits *= 0xc4ceb9fe1a85ec53ULL;
        bits ^= (bits >> 33);

        /* Write 16 hex chars */
        static const char hx[] = "0123456789abcdef";
        for (int j = 15; j >= 0 && pos < hex_len - 1; j--) {
            hex[pos++] = hx[(bits >> (j * 4)) & 0xF];
        }
    }
    hex[pos] = '\0';
    return hex;
}

double cs_fingerprint_compare(const char *fp1, const char *fp2) {
    if (!fp1 || !fp2) return 0.0;

    size_t len1 = strlen(fp1);
    size_t len2 = strlen(fp2);
    if (len1 == 0 && len2 == 0) return 1.0;

    size_t min_len = len1 < len2 ? len1 : len2;
    size_t max_len = len1 > len2 ? len1 : len2;
    if (max_len == 0) return 1.0;

    size_t matches = 0;
    for (size_t i = 0; i < min_len; i++)
        if (fp1[i] == fp2[i]) matches++;

    return (double)matches / (double)max_len;
}

#endif /* CS_IMPLEMENTATION */
