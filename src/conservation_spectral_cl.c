/*
 * conservation_spectral_cl.c — OpenCL host code for Conservation Spectral SDK
 *
 * Detects ALL OpenCL platforms/devices, auto-selects best GPU,
 * compiles kernels, and runs spectral graph analysis on GPU.
 */

/* CS_IMPLEMENTATION is NOT defined here — it's defined only in test_chord.c
 * which includes the full header-only implementation.
 * This file only uses the types and OpenCL functions. */
#include "conservation_spectral.h"
#include "conservation_spectral_cl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* ============================================================
 * Internal context structure
 * ============================================================ */

struct cscl_context {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       ctx;
    cl_command_queue queue;
    cl_program       program;

    /* Compiled kernels */
    cl_kernel k_build_adj_serial;
    cl_kernel k_laplacian_unnorm;
    cl_kernel k_laplacian_norm;
    cl_kernel k_shift_matrix;
    cl_kernel k_power_iter;
    cl_kernel k_deflate;
    cl_kernel k_cheeger;
    cl_kernel k_extract_fiedler;
    cl_kernel k_conservation_batch;
    cl_kernel k_anomaly_scores;

    char platform_name[256];
    char device_name[256];
};

/* ============================================================
 * Error helper
 * ============================================================ */

static const char *cl_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        default: return "UNKNOWN";
    }
}

#define CL_CHECK(call) do { \
    cl_int _err = (call); \
    if (_err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, cl_err_str(_err), _err); \
    } \
} while(0)

/* ============================================================
 * Kernel source loader
 * ============================================================ */

static char *load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, len, f) != (size_t)len) { /* best-effort */ }
    buf[len] = '\0';
    fclose(f);
    return buf;
}

static char *load_kernel_concat(const char *dir) {
    /* Load all .cl files from the kernel directory and concatenate */
    const char *files[] = {
        "laplacian.cl",
        "power_iteration.cl",
        "conservation.cl",
        "fiedler.cl",
        "anomaly.cl",
        NULL
    };

    size_t total = 1;
    char *sources[8];
    int count = 0;

    for (int i = 0; files[i] && count < 8; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, files[i]);
        sources[count] = load_file(path);
        if (sources[count]) {
            total += strlen(sources[count]) + 1;
            count++;
        }
    }

    if (count == 0) return NULL;

    char *combined = (char *)malloc(total);
    if (!combined) {
        for (int i = 0; i < count; i++) free(sources[i]);
        return NULL;
    }

    combined[0] = '\0';
    for (int i = 0; i < count; i++) {
        strcat(combined, sources[i]);
        free(sources[i]);
    }

    return combined;
}

/* ============================================================
 * Device selection
 * ============================================================ */

typedef struct {
    cl_platform_id platform;
    cl_device_id   device;
    cl_device_type type;
    int            score;  /* higher = better */
    char           platform_name[256];
    char           device_name[256];
} cscl_device_candidate;

static int score_device(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_GPU:          return 100;
        case CL_DEVICE_TYPE_ACCELERATOR:  return 80;
        case CL_DEVICE_TYPE_CPU:          return 50;
        default:                          return 10;
    }
}

static int select_best_device(cscl_device_candidate *best) {
    memset(best, 0, sizeof(*best));

    cl_uint n_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return -1;
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(n_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(n_platforms, platforms, NULL);

    best->score = -1;

    for (cl_uint p = 0; p < n_platforms; p++) {
        char pname[256] = {0};
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);

        cl_uint n_devices;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
        if (err != CL_SUCCESS || n_devices == 0) continue;

        cl_device_id *devices = (cl_device_id *)malloc(n_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, n_devices, devices, NULL);

        for (cl_uint d = 0; d < n_devices; d++) {
            cl_device_type dtype;
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);

            char dname[256] = {0};
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);

            int s = score_device(dtype);

            printf("  Found: %s / %s (type=0x%lx, score=%d)\n",
                   pname, dname, (unsigned long)dtype, s);

            if (s > best->score) {
                best->score = s;
                best->platform = platforms[p];
                best->device = devices[d];
                best->type = dtype;
                strncpy(best->platform_name, pname, 255);
                strncpy(best->device_name, dname, 255);
            }
        }

        free(devices);
    }

    free(platforms);
    return (best->score >= 0) ? 0 : -1;
}

/* ============================================================
 * Context creation / destruction
 * ============================================================ */

cscl_context *cscl_create(void) {
    cscl_device_candidate best;
    if (select_best_device(&best) != 0) {
        fprintf(stderr, "cscl_create: no suitable OpenCL device found\n");
        return NULL;
    }

    printf("Selected: %s / %s\n", best.platform_name, best.device_name);

    cscl_context *c = (cscl_context *)calloc(1, sizeof(cscl_context));
    if (!c) return NULL;

    c->platform = best.platform;
    c->device = best.device;
    strncpy(c->platform_name, best.platform_name, 255);
    strncpy(c->device_name, best.device_name, 255);

    cl_int err;

    /* Create context */
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)c->platform,
        0
    };
    c->ctx = clCreateContext(props, 1, &c->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context: %s\n", cl_err_str(err));
        free(c);
        return NULL;
    }

    /* Create command queue (compat: no properties for OpenCL 1.x) */
    c->queue = clCreateCommandQueue(c->ctx, c->device, 0, &err);
    /* Note: deprecated in OpenCL 2.0+ but widely available and compatible */
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create queue: %s\n", cl_err_str(err));
        clReleaseContext(c->ctx);
        free(c);
        return NULL;
    }

    /* Load and compile kernels */
    char *source = load_kernel_concat("kernels");
    if (!source) {
        /* Try relative to executable */
        source = load_kernel_concat("./kernels");
    }
    if (!source) {
        source = load_kernel_concat("../kernels");
    }
    if (!source) {
        fprintf(stderr, "Failed to load kernel sources\n");
        clReleaseCommandQueue(c->queue);
        clReleaseContext(c->ctx);
        free(c);
        return NULL;
    }

    const char *src_ptr = source;
    c->program = clCreateProgramWithSource(c->ctx, 1, &src_ptr, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program: %s\n", cl_err_str(err));
        free(source);
        clReleaseCommandQueue(c->queue);
        clReleaseContext(c->ctx);
        free(c);
        return NULL;
    }

    /* Build with math macros */
    err = clBuildProgram(c->program, 1, &c->device,
                         "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(c->program, c->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(c->program, c->device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Kernel build failed:\n%s\n", log);
        free(log);
        clReleaseProgram(c->program);
        free(source);
        clReleaseCommandQueue(c->queue);
        clReleaseContext(c->ctx);
        free(c);
        return NULL;
    }

    free(source);

    /* Extract kernels */
    c->k_build_adj_serial = clCreateKernel(c->program, "build_adjacency_serial", &err);
    c->k_laplacian_unnorm = clCreateKernel(c->program, "laplacian_unnormalized", &err);
    c->k_laplacian_norm   = clCreateKernel(c->program, "laplacian_normalized", &err);
    c->k_shift_matrix     = clCreateKernel(c->program, "shift_matrix", &err);
    c->k_power_iter       = clCreateKernel(c->program, "power_iteration_one", &err);
    c->k_deflate          = clCreateKernel(c->program, "deflate", &err);
    c->k_cheeger          = clCreateKernel(c->program, "cheeger_constant", &err);
    c->k_extract_fiedler  = clCreateKernel(c->program, "extract_fiedler", &err);
    c->k_conservation_batch = clCreateKernel(c->program, "conservation_ratio_batch", &err);
    c->k_anomaly_scores   = clCreateKernel(c->program, "anomaly_scores", &err);

    printf("OpenCL context ready: %s / %s\n", c->platform_name, c->device_name);
    return c;
}

void cscl_free(cscl_context *ctx) {
    if (!ctx) return;

    if (ctx->k_build_adj_serial) clReleaseKernel(ctx->k_build_adj_serial);
    if (ctx->k_laplacian_unnorm) clReleaseKernel(ctx->k_laplacian_unnorm);
    if (ctx->k_laplacian_norm)   clReleaseKernel(ctx->k_laplacian_norm);
    if (ctx->k_shift_matrix)     clReleaseKernel(ctx->k_shift_matrix);
    if (ctx->k_power_iter)       clReleaseKernel(ctx->k_power_iter);
    if (ctx->k_deflate)          clReleaseKernel(ctx->k_deflate);
    if (ctx->k_cheeger)          clReleaseKernel(ctx->k_cheeger);
    if (ctx->k_extract_fiedler)  clReleaseKernel(ctx->k_extract_fiedler);
    if (ctx->k_conservation_batch) clReleaseKernel(ctx->k_conservation_batch);
    if (ctx->k_anomaly_scores)   clReleaseKernel(ctx->k_anomaly_scores);

    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue)   clReleaseCommandQueue(ctx->queue);
    if (ctx->ctx)     clReleaseContext(ctx->ctx);

    free(ctx);
}

void cscl_print_info(const cscl_context *ctx) {
    if (!ctx) { printf("No OpenCL context.\n"); return; }
    printf("Platform: %s\n", ctx->platform_name);
    printf("Device:   %s\n", ctx->device_name);

    cl_uint compute_units;
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(compute_units), &compute_units, NULL);

    cl_ulong global_mem;
    clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(global_mem), &global_mem, NULL);

    cl_uint max_freq;
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(max_freq), &max_freq, NULL);

    printf("Compute units: %u\n", compute_units);
    printf("Global memory: %.0f MB\n", global_mem / (1024.0 * 1024.0));
    printf("Max frequency: %u MHz\n", max_freq);
}

/* ============================================================
 * GPU Laplacian
 * ============================================================ */

cs_laplacian cscl_build_laplacian(const cscl_context *ctx,
                                   const cs_graph *g, bool normalized) {
    cs_laplacian lap;
    memset(&lap, 0, sizeof(lap));
    if (!ctx || !g) return lap;

    size_t n = g->n_vertices;
    size_t nn = n * n;
    lap.n = n;
    lap.normalized = normalized;

    cl_int err;
    size_t n_edges = g->n_edges;

    /* Prepare edge data */
    cl_uint *edge_from = (cl_uint *)calloc(n_edges, sizeof(cl_uint));
    cl_uint *edge_to   = (cl_uint *)calloc(n_edges, sizeof(cl_uint));
    double *edge_wt = (double *)calloc(n_edges, sizeof(double));
    for (size_t e = 0; e < n_edges; e++) {
        edge_from[e] = (cl_uint)g->edges[e].from;
        edge_to[e]   = (cl_uint)g->edges[e].to;
        edge_wt[e]   = g->edges[e].weight;
    }

    /* Allocate GPU buffers */
    cl_mem d_from = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n_edges * sizeof(cl_uint), edge_from, &err);
    cl_mem d_to   = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n_edges * sizeof(cl_uint), edge_to, &err);
    cl_mem d_wt   = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n_edges * sizeof(double), edge_wt, &err);

    double *zeros = (double *)calloc(nn, sizeof(double));
    cl_mem d_W = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 nn * sizeof(double), zeros, &err);
    free(zeros);

    /* Build adjacency: build_adjacency_serial(from, to, weight, W, n, n_edges) */
    clSetKernelArg(ctx->k_build_adj_serial, 0, sizeof(cl_mem), &d_from);
    clSetKernelArg(ctx->k_build_adj_serial, 1, sizeof(cl_mem), &d_to);
    clSetKernelArg(ctx->k_build_adj_serial, 2, sizeof(cl_mem), &d_wt);
    clSetKernelArg(ctx->k_build_adj_serial, 3, sizeof(cl_mem), &d_W);
    clSetKernelArg(ctx->k_build_adj_serial, 4, sizeof(cl_uint), &(cl_uint){(cl_uint)n});
    clSetKernelArg(ctx->k_build_adj_serial, 5, sizeof(cl_uint), &(cl_uint){(cl_uint)n_edges});

    size_t work_size = 1;
    clEnqueueNDRangeKernel(ctx->queue, ctx->k_build_adj_serial, 1, NULL,
                            &work_size, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);

    /* Build Laplacian */
    cl_mem d_L = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE,
                                 nn * sizeof(double), NULL, &err);

    cl_kernel k_lap = normalized ? ctx->k_laplacian_norm : ctx->k_laplacian_unnorm;
    clSetKernelArg(k_lap, 0, sizeof(cl_mem), &d_W);
    clSetKernelArg(k_lap, 1, sizeof(cl_mem), &d_L);
    clSetKernelArg(k_lap, 2, sizeof(cl_uint), &(cl_uint){(cl_uint)n});

    size_t lap_work = nn;
    clEnqueueNDRangeKernel(ctx->queue, k_lap, 1, NULL,
                            &lap_work, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);

    /* Read back */
    lap.values = (double *)calloc(nn, sizeof(double));
    clEnqueueReadBuffer(ctx->queue, d_L, CL_TRUE, 0, nn * sizeof(double),
                         lap.values, 0, NULL, NULL);

    /* Cleanup GPU buffers */
    clReleaseMemObject(d_from);
    clReleaseMemObject(d_to);
    clReleaseMemObject(d_wt);
    clReleaseMemObject(d_W);
    clReleaseMemObject(d_L);

    free(edge_from);
    free(edge_to);
    free(edge_wt);

    return lap;
}

/* ============================================================
 * GPU Eigen decomposition
 * ============================================================ */

cs_eigen cscl_eigendecompose(const cscl_context *ctx,
                              const cs_laplacian *lap, size_t k) {
    cs_eigen eig;
    memset(&eig, 0, sizeof(eig));
    if (!ctx || !lap) return eig;

    size_t n = lap->n;
    size_t nn = n * n;
    if (k == 0 || k > n) k = n;
    eig.n = n;

    cl_int err;

    /* Find shift = max diagonal of L */
    double shift = 0.0;
    for (size_t i = 0; i < n; i++)
        if (lap->values[i * n + i] > shift)
            shift = lap->values[i * n + i];

    /* Allocate output arrays */
    eig.eigenvalues  = (double *)calloc(n, sizeof(double));
    eig.eigenvectors = (double *)calloc(nn, sizeof(double));
    if (!eig.eigenvalues || !eig.eigenvectors) return eig;

    /* Upload Laplacian to GPU */
    cl_mem d_L = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 nn * sizeof(double), lap->values, &err);

    /* Build M = shift*I - L on GPU */
    cl_mem d_M = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE,
                                 nn * sizeof(double), NULL, &err);
    clSetKernelArg(ctx->k_shift_matrix, 0, sizeof(cl_mem), &d_L);
    clSetKernelArg(ctx->k_shift_matrix, 1, sizeof(cl_mem), &d_M);
    clSetKernelArg(ctx->k_shift_matrix, 2, sizeof(cl_double), &(cl_double){shift});
    clSetKernelArg(ctx->k_shift_matrix, 3, sizeof(cl_uint), &(cl_uint){(cl_uint)n});

    size_t shift_work = nn;
    clEnqueueNDRangeKernel(ctx->queue, ctx->k_shift_matrix, 1, NULL,
                            &shift_work, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);

    /* Residual matrix R (starts as copy of M, gets deflated) */
    double *R_host = (double *)malloc(nn * sizeof(double));
    clEnqueueReadBuffer(ctx->queue, d_M, CL_TRUE, 0, nn * sizeof(double),
                         R_host, 0, NULL, NULL);

    /* Iterate: for each eigenvalue, run power iteration on GPU, then deflate */
    cl_mem d_R = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE,
                                 nn * sizeof(double), NULL, &err);
    cl_mem d_v = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE,
                                 n * sizeof(double), NULL, &err);
    cl_mem d_w = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE,
                                 n * sizeof(double), NULL, &err);
    cl_mem d_lambda = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE,
                                      sizeof(double), NULL, &err);

    for (size_t ev = 0; ev < k; ev++) {
        /* Upload current R */
        clEnqueueWriteBuffer(ctx->queue, d_R, CL_TRUE, 0, nn * sizeof(double),
                              R_host, 0, NULL, NULL);

        /* Run power iteration on GPU */
        clSetKernelArg(ctx->k_power_iter, 0, sizeof(cl_mem), &d_R);
        clSetKernelArg(ctx->k_power_iter, 1, sizeof(cl_mem), &d_v);
        clSetKernelArg(ctx->k_power_iter, 2, sizeof(cl_mem), &d_w);
        clSetKernelArg(ctx->k_power_iter, 3, sizeof(cl_mem), &d_lambda);
        clSetKernelArg(ctx->k_power_iter, 4, sizeof(cl_uint), &(cl_uint){(cl_uint)n});
        clSetKernelArg(ctx->k_power_iter, 5, sizeof(cl_uint), &(cl_uint){2000});
        clSetKernelArg(ctx->k_power_iter, 6, sizeof(cl_uint), &(cl_uint){(cl_uint)ev});

        size_t pi_work = 1;
        clEnqueueNDRangeKernel(ctx->queue, ctx->k_power_iter, 1, NULL,
                                &pi_work, NULL, 0, NULL, NULL);
        clFinish(ctx->queue);

        /* Read back eigenvalue and eigenvector */
        double lambda_M;
        clEnqueueReadBuffer(ctx->queue, d_lambda, CL_TRUE, 0, sizeof(double),
                             &lambda_M, 0, NULL, NULL);

        double *v_host = (double *)malloc(n * sizeof(double));
        clEnqueueReadBuffer(ctx->queue, d_v, CL_TRUE, 0, n * sizeof(double),
                             v_host, 0, NULL, NULL);

        /* Store: eigenvalue of L = shift - lambda_M */
        eig.eigenvalues[ev] = shift - lambda_M;

        /* Store eigenvector (column ev, column-major) */
        for (size_t i = 0; i < n; i++)
            eig.eigenvectors[ev * n + i] = v_host[i];

        /* Deflate on host: R = R - lambda_M * v*v^T */
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                R_host[i * n + j] -= lambda_M * v_host[i] * v_host[j];

        free(v_host);
    }

    clReleaseMemObject(d_R);
    clReleaseMemObject(d_v);
    clReleaseMemObject(d_w);
    clReleaseMemObject(d_lambda);
    clReleaseMemObject(d_M);
    clReleaseMemObject(d_L);
    free(R_host);

    /* Sort eigenvalues ascending (with corresponding eigenvectors) */
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (eig.eigenvalues[j] < eig.eigenvalues[i]) {
                double tmp = eig.eigenvalues[i];
                eig.eigenvalues[i] = eig.eigenvalues[j];
                eig.eigenvalues[j] = tmp;
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

/* ============================================================
 * GPU conservation ratio
 * ============================================================ */

double cscl_conservation_ratio(const cscl_context *ctx,
                                const cs_eigen *eigen, const double *attr,
                                size_t attr_len, size_t eigenvector_index) {
    if (!ctx || !eigen || !attr || eigen->n != attr_len) return -1.0;
    /* For single-ratio query, fall back to CPU (GPU overhead not worth it) */
    return cs_conservation_ratio(eigen, attr, attr_len, eigenvector_index);
}

/* ============================================================
 * GPU Fiedler vector extraction
 * ============================================================ */

void cscl_fiedler(const cscl_context *ctx, const cs_eigen *eigen,
                  double *out_vector, size_t n) {
    if (!ctx || !eigen || !out_vector || eigen->n < 2) return;

    cl_int err;
    size_t nn = n * n;

    cl_mem d_ev = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  nn * sizeof(double), eigen->eigenvectors, &err);
    cl_mem d_fiedler = clCreateBuffer(ctx->ctx, CL_MEM_WRITE_ONLY,
                                       n * sizeof(double), NULL, &err);

    clSetKernelArg(ctx->k_extract_fiedler, 0, sizeof(cl_mem), &d_ev);
    clSetKernelArg(ctx->k_extract_fiedler, 1, sizeof(cl_mem), &d_fiedler);
    clSetKernelArg(ctx->k_extract_fiedler, 2, sizeof(cl_uint), &(cl_uint){(cl_uint)n});

    size_t f_work = n;
    clEnqueueNDRangeKernel(ctx->queue, ctx->k_extract_fiedler, 1, NULL,
                            &f_work, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(ctx->queue, d_fiedler, CL_TRUE, 0, n * sizeof(double),
                         out_vector, 0, NULL, NULL);

    clReleaseMemObject(d_ev);
    clReleaseMemObject(d_fiedler);
}

/* ============================================================
 * GPU anomaly detection
 * ============================================================ */

void cscl_detect_anomalies(const cscl_context *ctx,
                            const cs_graph *g, const cs_eigen *eigen,
                            cs_anomaly *out_anomalies, size_t *out_count) {
    if (!ctx || !g || !eigen || !out_anomalies || !out_count) {
        if (out_count) *out_count = 0;
        return;
    }

    size_t n = eigen->n;
    size_t nn = n * n;
    *out_count = 0;

    /* Extract attributes */
    double *attr = (double *)calloc(n, sizeof(double));
    for (size_t i = 0; i < n; i++)
        attr[i] = g->vertices[i].attribute;

    /* Get Fiedler vector */
    double *fiedler = (double *)calloc(n, sizeof(double));
    cscl_fiedler(ctx, eigen, fiedler, n);

    /* Upload to GPU */
    cl_int err;
    cl_mem d_fiedler = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       n * sizeof(double), fiedler, &err);
    cl_mem d_attr = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n * sizeof(double), attr, &err);
    cl_mem d_ev = clCreateBuffer(ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  nn * sizeof(double), eigen->eigenvectors, &err);
    cl_mem d_scores = clCreateBuffer(ctx->ctx, CL_MEM_WRITE_ONLY,
                                      n * sizeof(double), NULL, &err);

    /* Run anomaly scoring kernel */
    clSetKernelArg(ctx->k_anomaly_scores, 0, sizeof(cl_mem), &d_fiedler);
    clSetKernelArg(ctx->k_anomaly_scores, 1, sizeof(cl_mem), &d_attr);
    clSetKernelArg(ctx->k_anomaly_scores, 2, sizeof(cl_mem), &d_ev);
    clSetKernelArg(ctx->k_anomaly_scores, 3, sizeof(cl_mem), &d_scores);
    clSetKernelArg(ctx->k_anomaly_scores, 4, sizeof(cl_uint), &(cl_uint){(cl_uint)n});

    size_t a_work = n;
    clEnqueueNDRangeKernel(ctx->queue, ctx->k_anomaly_scores, 1, NULL,
                            &a_work, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);

    /* Read back scores */
    double *scores = (double *)calloc(n, sizeof(double));
    clEnqueueReadBuffer(ctx->queue, d_scores, CL_TRUE, 0, n * sizeof(double),
                         scores, 0, NULL, NULL);

    /* Compute threshold: mean + 2*std */
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) mean += scores[i];
    mean /= (double)n;

    double std = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = scores[i] - mean;
        std += d * d;
    }
    std = sqrt(std / (double)n);

    double threshold = mean + 2.0 * std;

    /* Collect anomalies above threshold */
    size_t count = 0;
    for (size_t i = 0; i < n && count < 10; i++) {
        if (scores[i] > threshold) {
            out_anomalies[count].vertex_id = g->vertices[i].id;
            out_anomalies[count].eigenvector_index = 1;
            out_anomalies[count].score = scores[i];
            snprintf(out_anomalies[count].suggested_fix, 256,
                     "Vertex %llu has anomalous attribute %.4f for its spectral community (score=%.4f)",
                     (unsigned long long)g->vertices[i].id, attr[i], scores[i]);
            count++;
        }
    }
    *out_count = count;

    clReleaseMemObject(d_fiedler);
    clReleaseMemObject(d_attr);
    clReleaseMemObject(d_ev);
    clReleaseMemObject(d_scores);
    free(attr);
    free(fiedler);
    free(scores);
}

/* ============================================================
 * Full GPU analysis
 * ============================================================ */

cs_report cscl_analyze(const cscl_context *ctx,
                        const cs_graph *g, const char *attribute_name) {
    cs_report rpt;
    memset(&rpt, 0, sizeof(rpt));
    if (!ctx || !g) return rpt;

    cs_laplacian lap = cscl_build_laplacian(ctx, g, false);
    if (!lap.values) return rpt;

    cs_eigen eig = cscl_eigendecompose(ctx, &lap, 0);
    if (!eig.eigenvalues) { cs_laplacian_free(&lap); return rpt; }

    /* Collect attributes */
    double *attr = (double *)calloc(g->n_vertices, sizeof(double));
    for (size_t i = 0; i < g->n_vertices; i++)
        attr[i] = g->vertices[i].attribute;

    /* Compute ratios (use CPU for batch of small n — GPU overhead not worth it) */
    rpt.n_ratios = eig.n;
    rpt.ratios = (cs_ratio *)calloc(rpt.n_ratios, sizeof(cs_ratio));
    for (size_t k = 0; k < rpt.n_ratios; k++) {
        rpt.ratios[k].eigenvector_index = k;
        rpt.ratios[k].eigenvalue = eig.eigenvalues[k];
        rpt.ratios[k].ratio = cs_conservation_ratio(&eig, attr, g->n_vertices, k);
    }

    rpt.spectral_gap = cs_spectral_gap(&eig);

    /* Cheeger constant from Fiedler vector */
    if (eig.n >= 2) {
        double *fiedler = &eig.eigenvectors[1 * eig.n];
        rpt.cheeger_constant = cs_cheeger_constant(&lap, fiedler);
    }

    free(attr);
    cs_laplacian_free(&lap);
    cs_eigen_free(&eig);
    return rpt;
}
