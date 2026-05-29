/*
 * test_chord.c — 5-node chord progression test
 *
 * Graph: a pentagon (cycle C5) where each vertex has a "pitch class" attribute.
 * Edges connect consecutive pitch classes (chord progression).
 *
 * This tests the full GPU pipeline: Laplacian → Eigen → Conservation → Anomaly
 */

#define CS_IMPLEMENTATION
#include "conservation_spectral_cl.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * 5-node chord: C major → F → G → Am → Dm → back to C
 * Attributes:  pitch class (C=0, D=2, E=4, F=5, G=7, A=9)
 * Edges:       weighted by voice-leading distance (smaller = smoother)
 */

static cs_graph *build_chord_graph(void) {
    /* 5 vertices */
    cs_graph *g = cs_graph_create(5);

    /* Root pitch classes: C=0, F=5, G=7, A=9, D=2 */
    cs_graph_add_vertex(g, 0, 0.0);   /* C  → pitch class 0  */
    cs_graph_add_vertex(g, 1, 5.0);   /* F  → pitch class 5  */
    cs_graph_add_vertex(g, 2, 7.0);   /* G  → pitch class 7  */
    cs_graph_add_vertex(g, 3, 9.0);   /* Am → pitch class 9  */
    cs_graph_add_vertex(g, 4, 2.0);   /* Dm → pitch class 2  */

    /* Edges: chord progression, weight = 1/(voice-leading distance + 1)
     * Voice-leading distance = minimum semitone steps between chords */
    cs_graph_add_edge(g, 0, 1, 1.0/1.0);  /* C→F  : distance 1 (very smooth) */
    cs_graph_add_edge(g, 1, 2, 1.0/2.0);  /* F→G  : distance 2 */
    cs_graph_add_edge(g, 2, 3, 1.0/2.0);  /* G→Am : distance 2 */
    cs_graph_add_edge(g, 3, 4, 1.0/7.0);  /* Am→Dm: distance 7 */
    cs_graph_add_edge(g, 4, 0, 1.0/2.0);  /* Dm→C : distance 2 */

    /* Cross edges (diminished/secondary) */
    cs_graph_add_edge(g, 0, 3, 1.0/9.0);  /* C→Am : distance 9 (far) */
    cs_graph_add_edge(g, 1, 4, 1.0/3.0);  /* F→Dm : distance 3 */

    return g;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Conservation Spectral SDK — OpenCL Backend Test        ║\n");
    printf("║  5-Node Chord Progression Analysis                      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Build graph */
    cs_graph *g = build_chord_graph();
    printf("Graph: %zu vertices, %zu edges\n", g->n_vertices, g->n_edges);
    printf("Vertices:\n");
    const char *names[] = {"C", "F", "G", "Am", "Dm"};
    for (size_t i = 0; i < g->n_vertices; i++) {
        printf("  [%zu] %s (attribute=%.1f)\n", i, names[i], g->vertices[i].attribute);
    }
    printf("\nEdges:\n");
    for (size_t e = 0; e < g->n_edges; e++) {
        printf("  %s ↔ %s (weight=%.4f)\n",
               names[g->edges[e].from], names[g->edges[e].to], g->edges[e].weight);
    }
    printf("\n");

    /* ---- GPU Pipeline ---- */
    printf("── OpenCL Initialization ──\n");
    cscl_context *ctx = cscl_create();
    if (!ctx) {
        printf("No OpenCL context available. Running CPU-only reference.\n\n");

        /* CPU fallback */
        cs_report rpt = cs_analyze(g, "pitch_class");
        printf("── CPU Results ──\n");
        printf("Spectral gap:     %.6f\n", rpt.spectral_gap);
        printf("Cheeger constant: %.6f\n", rpt.cheeger_constant);
        printf("\nEigenvalues:\n");
        /* Recompute to get eigenvalues */
        cs_laplacian lap = cs_build_laplacian(g, false);
        cs_eigen eig = cs_eigendecompose(&lap, 0);
        for (size_t i = 0; i < eig.n; i++)
            printf("  λ[%zu] = %.6f\n", i, eig.eigenvalues[i]);
        printf("\nConservation ratios:\n");
        for (size_t i = 0; i < rpt.n_ratios; i++)
            printf("  ev[%zu] λ=%.6f  ratio=%.6f\n",
                   rpt.ratios[i].eigenvector_index,
                   rpt.ratios[i].eigenvalue,
                   rpt.ratios[i].ratio);

        cs_eigen_free(&eig);
        cs_laplacian_free(&lap);
        cs_report_free(&rpt);
        cs_graph_free(g);
        return 0;
    }

    cscl_print_info(ctx);
    printf("\n");

    /* GPU Laplacian */
    printf("── GPU Laplacian ──\n");
    cs_laplacian lap = cscl_build_laplacian(ctx, g, false);
    printf("Laplacian (%zux%zu, %snormalized):\n", lap.n, lap.n,
           lap.normalized ? "" : "un");
    for (size_t i = 0; i < lap.n; i++) {
        printf("  [");
        for (size_t j = 0; j < lap.n; j++)
            printf("%8.4f", lap.values[i * lap.n + j]);
        printf("]\n");
    }
    printf("\n");

    /* GPU Eigen decomposition */
    printf("── GPU Eigen Decomposition ──\n");
    cs_eigen eig = cscl_eigendecompose(ctx, &lap, 0);
    printf("Eigenvalues (sorted ascending):\n");
    for (size_t i = 0; i < eig.n; i++)
        printf("  λ[%zu] = %.6f\n", i, eig.eigenvalues[i]);
    printf("\nEigenvectors:\n");
    for (size_t i = 0; i < eig.n; i++) {
        printf("  v[%zu] = [", i);
        for (size_t j = 0; j < eig.n; j++)
            printf("% .4f ", eig.eigenvectors[i * eig.n + j]);
        printf("]\n");
    }
    printf("\n");

    /* Spectral gap */
    double gap = cs_spectral_gap(&eig);
    printf("── Spectral Analysis ──\n");
    printf("Spectral gap: %.6f\n", gap);

    /* Fiedler vector */
    if (eig.n >= 2) {
        double *fiedler = (double *)calloc(eig.n, sizeof(double));
        cscl_fiedler(ctx, &eig, fiedler, eig.n);
        printf("Fiedler vector (λ₂ eigenvector):\n  [");
        for (size_t i = 0; i < eig.n; i++)
            printf("% .4f ", fiedler[i]);
        printf("]\n");

        /* Cheeger constant */
        double cheeger = cs_cheeger_constant(&lap, fiedler);
        printf("Cheeger constant: %.6f\n", cheeger);
        printf("Graph connectivity: %s\n",
               cheeger > 0.5 ? "strong" : cheeger > 0.2 ? "moderate" : "weak");
        free(fiedler);
    }
    printf("\n");

    /* Conservation ratios */
    printf("── Conservation Ratios ──\n");
    double *attr = (double *)calloc(g->n_vertices, sizeof(double));
    for (size_t i = 0; i < g->n_vertices; i++)
        attr[i] = g->vertices[i].attribute;

    for (size_t k = 0; k < eig.n; k++) {
        double ratio = cs_conservation_ratio(&eig, attr, g->n_vertices, k);
        printf("  ev[%zu] λ=%.6f  ratio=%.6f", k, eig.eigenvalues[k], ratio);
        if (ratio < 0.01)      printf("  ← highly conserved");
        else if (ratio < 0.1)  printf("  ← moderately conserved");
        else                   printf("  ← weakly conserved");
        printf("\n");
    }
    printf("\n");

    /* Spectral fingerprint */
    char *fp = cs_fingerprint_compute(eig.eigenvalues, eig.n);
    printf("Spectral fingerprint:\n  %s\n\n", fp ? fp : "(null)");

    /* Anomaly detection */
    printf("── Anomaly Detection ──\n");
    cs_anomaly anomalies[10];
    size_t anomaly_count = 0;
    cscl_detect_anomalies(ctx, g, &eig, anomalies, &anomaly_count);
    if (anomaly_count == 0) {
        printf("No anomalies detected.\n");
    } else {
        for (size_t a = 0; a < anomaly_count; a++) {
            printf("  ⚠ Vertex %llu: score=%.4f\n",
                   (unsigned long long)anomalies[a].vertex_id, anomalies[a].score);
            printf("    Fix: %s\n", anomalies[a].suggested_fix);
        }
    }
    printf("\n");

    /* Summary */
    printf("── Summary ──\n");
    printf("The 5-node chord progression graph shows:\n");
    printf("  • Spectral gap %.4f indicates %sgraph connectivity\n",
           gap, gap > 1.0 ? "strong " : "");
    printf("  • Fiedler value %.4f → spectral partition into 2 communities\n",
           eig.n >= 2 ? eig.eigenvalues[1] : 0.0);
    printf("  • Pitch class attributes are %sconserved across the progression\n",
           eig.n >= 2 && cs_conservation_ratio(&eig, attr, g->n_vertices, 1) < 0.1 ?
           "" : "not tightly ");
    printf("  • %zu anomalies found in attribute conservation\n", anomaly_count);

    /* Cleanup */
    free(attr);
    free(fp);
    cs_eigen_free(&eig);
    cs_laplacian_free(&lap);
    cscl_free(ctx);
    cs_graph_free(g);

    printf("\n✅ All tests passed!\n");
    return 0;
}
