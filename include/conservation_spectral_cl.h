/*
 * conservation_spectral_cl.h — Conservation Spectral SDK (OpenCL backend)
 *
 * Include conservation_spectral.h (with CS_IMPLEMENTATION in one .c file)
 * before this header.
 *
 * Version: 0.1.0
 */

#ifndef CONSERVATION_SPECTRAL_CL_H
#define CONSERVATION_SPECTRAL_CL_H

#include "conservation_spectral.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * OpenCL context
 * ============================================================ */

typedef struct cscl_context cscl_context;

cscl_context *cscl_create(void);
void cscl_free(cscl_context *ctx);
void cscl_print_info(const cscl_context *ctx);

/* ============================================================
 * GPU-accelerated operations
 * ============================================================ */

cs_laplacian cscl_build_laplacian(const cscl_context *ctx,
                                   const cs_graph *g, bool normalized);

cs_eigen cscl_eigendecompose(const cscl_context *ctx,
                              const cs_laplacian *lap, size_t k);

cs_report cscl_analyze(const cscl_context *ctx,
                        const cs_graph *g, const char *attribute_name);

double cscl_conservation_ratio(const cscl_context *ctx,
                                const cs_eigen *eigen, const double *attr,
                                size_t attr_len, size_t eigenvector_index);

void cscl_fiedler(const cscl_context *ctx, const cs_eigen *eigen,
                  double *out_vector, size_t n);

void cscl_detect_anomalies(const cscl_context *ctx,
                            const cs_graph *g, const cs_eigen *eigen,
                            cs_anomaly *out_anomalies, size_t *out_count);

#ifdef __cplusplus
}
#endif

#endif /* CONSERVATION_SPECTRAL_CL_H */
