/*
 * simulation/c/seir_core.c
 * ========================
 * Pure-C implementation of the SEIR-V-D RK4 stepper + commuter-coupled FoI.
 *
 * Matches the semantics of simulation/sim/stepper.py::rk4_step_jit exactly
 * (deterministic RK4, 6 compartments S E I R V D, FoI via row-stochastic
 * mobility matrix). Built as a shared library and loaded from Python via
 * ctypes — no pybind11, no Cython, no external dependencies beyond libc
 * and libm.
 *
 * Build:
 *   cc -O3 -shared -fPIC -o seir_core.dylib seir_core.c -lm   (macOS/Linux; B-P3: no -ffast-math/-march=native — IEEE + reproducibility)
 *   cl /O2 /LD seir_core.c                                                               (Windows MSVC)
 *
 * Python loader:
 *   from simulation.sim.stepper_c import rk4_step_c   (auto-loads .dylib/.so/.dll)
 *
 * ABI contract (all doubles, row-major, contiguous):
 *   void rk4_step_c(
 *       const double *state_in,   // shape (G, 6)
 *       double       *state_out,  // shape (G, 6), caller-allocated
 *       int           G,
 *       double        dt,
 *       double        beta, sigma, gamma, omega, VE, V_waning, ifr,
 *       const double *vax_rate,   // shape (G,)
 *       const double *populations,// shape (G,)  (unused but kept for parity)
 *       const double *mobility,   // shape (G, G) row-major
 *       const double *daytime_pop // shape (G,)
 *   );
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Small stack scratch for typical G ≤ 64. Fall back to heap when G is
 * larger (e.g. future 3500-dong extension). */
#define SEIR_STACK_G 64

static inline void
seirvd_rhs_c(const double *state, double *rhs, int G,
             double beta, double sigma, double gam, double omega,
             double VE, double V_waning, double ifr,
             const double *vax_rate,
             const double *mobility, const double *daytime_pop)
{
    /* Allocate scratch for FoI intermediates (I_present, prev_j, lam). */
    double lam_buf_stack[3 * SEIR_STACK_G];
    double *buf = lam_buf_stack;
    double *heap = NULL;
    if (G > SEIR_STACK_G) {
        heap = (double *)malloc((size_t)(3 * G) * sizeof(double));
        if (!heap) { /* caller sees zero rhs rather than crash */
            memset(rhs, 0, (size_t)G * 6 * sizeof(double));
            return;
        }
        buf = heap;
    }
    double *I_present = buf;
    double *prev_j    = buf + G;
    double *lam       = buf + 2 * G;

    /* I_present[j] = sum_i mobility[i, j] * I[i] where I = state[:, 2] */
    for (int j = 0; j < G; ++j) {
        double s = 0.0;
        for (int i = 0; i < G; ++i) {
            s += mobility[i * G + j] * state[i * 6 + 2];
        }
        I_present[j] = s;
    }

    /* prev_j[j] = I_present[j] / max(daytime_pop[j], 1) */
    for (int j = 0; j < G; ++j) {
        double d = daytime_pop[j];
        if (d < 1.0) d = 1.0;
        prev_j[j] = I_present[j] / d;
    }

    /* lam[i] = beta * sum_j mobility[i, j] * prev_j[j], clipped ≥ 0 */
    for (int i = 0; i < G; ++i) {
        double s = 0.0;
        const double *row = &mobility[i * G];
        for (int j = 0; j < G; ++j) {
            s += row[j] * prev_j[j];
        }
        double v = beta * s;
        lam[i] = v > 0.0 ? v : 0.0;
    }

    /* SEIR-V-D RHS */
    for (int i = 0; i < G; ++i) {
        const double S = state[i * 6 + 0];
        const double E = state[i * 6 + 1];
        const double I = state[i * 6 + 2];
        const double R = state[i * 6 + 3];
        const double V = state[i * 6 + 4];
        const double lm  = lam[i];
        const double lmv = (1.0 - VE) * lm;
        const double vr  = vax_rate[i];

        rhs[i * 6 + 0] = -lm * S - vr * S + omega * R + V_waning * V;
        rhs[i * 6 + 1] = lm * S + lmv * V - sigma * E;
        rhs[i * 6 + 2] = sigma * E - gam * I;
        rhs[i * 6 + 3] = gam * (1.0 - ifr) * I - omega * R;
        rhs[i * 6 + 4] = vr * S - lmv * V - V_waning * V;
        rhs[i * 6 + 5] = gam * ifr * I;
    }

    if (heap) free(heap);
}

/* Exported symbol — visible to ctypes.CDLL. */
#if defined(_WIN32)
#define SEIR_EXPORT __declspec(dllexport)
#else
#define SEIR_EXPORT __attribute__((visibility("default")))
#endif

SEIR_EXPORT void
rk4_step_c(const double *state_in, double *state_out, int G, double dt,
           double beta, double sigma, double gam, double omega,
           double VE, double V_waning, double ifr,
           const double *vax_rate,
           const double *populations,
           const double *mobility,
           const double *daytime_pop)
{
    (void)populations; /* parity with Numba signature; unused here */
    const int NC = 6;
    const size_t N = (size_t)G * NC;

    /* k1, k2, k3, k4 + tmp state for mid-step evaluations */
    double k1_stack[SEIR_STACK_G * 6];
    double k2_stack[SEIR_STACK_G * 6];
    double k3_stack[SEIR_STACK_G * 6];
    double k4_stack[SEIR_STACK_G * 6];
    double tmp_stack[SEIR_STACK_G * 6];
    double *k1 = k1_stack, *k2 = k2_stack, *k3 = k3_stack, *k4 = k4_stack, *tmp = tmp_stack;
    double *heap = NULL;
    if (G > SEIR_STACK_G) {
        heap = (double *)malloc(5 * N * sizeof(double));
        if (!heap) {
            memcpy(state_out, state_in, N * sizeof(double));
            return;
        }
        k1  = heap;
        k2  = heap + N;
        k3  = heap + 2 * N;
        k4  = heap + 3 * N;
        tmp = heap + 4 * N;
    }

    /* k1 = f(y) */
    seirvd_rhs_c(state_in, k1, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                 vax_rate, mobility, daytime_pop);
    /* tmp = y + 0.5 dt k1 */
    for (size_t n = 0; n < N; ++n) tmp[n] = state_in[n] + 0.5 * dt * k1[n];
    seirvd_rhs_c(tmp, k2, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                 vax_rate, mobility, daytime_pop);
    /* tmp = y + 0.5 dt k2 */
    for (size_t n = 0; n < N; ++n) tmp[n] = state_in[n] + 0.5 * dt * k2[n];
    seirvd_rhs_c(tmp, k3, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                 vax_rate, mobility, daytime_pop);
    /* tmp = y + dt k3 */
    for (size_t n = 0; n < N; ++n) tmp[n] = state_in[n] + dt * k3[n];
    seirvd_rhs_c(tmp, k4, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                 vax_rate, mobility, daytime_pop);

    /* state_out = y + dt/6 (k1 + 2 k2 + 2 k3 + k4), clamped ≥ 0 */
    const double s6 = dt / 6.0;
    for (size_t n = 0; n < N; ++n) {
        double v = state_in[n] + s6 * (k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n]);
        state_out[n] = v > 0.0 ? v : 0.0;
    }

    if (heap) free(heap);
}

/* Batched N-step RK4 — runs `n_steps` sub-steps inside the C kernel so the
 * Python ↔ C boundary is crossed only once per batch. This is where C wins
 * over Numba: no per-call ctypes overhead across n_steps iterations.
 *
 * state_in is NOT mutated; state_out ends with the final post-step state.
 * Memory allocation happens once inside this function. */
SEIR_EXPORT void
rk4_step_batch_c(const double *state_in, double *state_out, int G, int n_steps,
                 double dt, double beta, double sigma, double gam, double omega,
                 double VE, double V_waning, double ifr,
                 const double *vax_rate,
                 const double *populations,
                 const double *mobility,
                 const double *daytime_pop)
{
    const size_t N = (size_t)G * 6;

    /* Copy initial state into state_out (we will iterate in-place there). */
    memcpy(state_out, state_in, N * sizeof(double));

    /* Allocate scratch once outside the step loop. */
    double *k1  = (double *)malloc(N * sizeof(double));
    double *k2  = (double *)malloc(N * sizeof(double));
    double *k3  = (double *)malloc(N * sizeof(double));
    double *k4  = (double *)malloc(N * sizeof(double));
    double *tmp = (double *)malloc(N * sizeof(double));
    if (!k1 || !k2 || !k3 || !k4 || !tmp) {
        free(k1); free(k2); free(k3); free(k4); free(tmp);
        return;
    }

    for (int step = 0; step < n_steps; ++step) {
        seirvd_rhs_c(state_out, k1, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                     vax_rate, mobility, daytime_pop);
        for (size_t n = 0; n < N; ++n) tmp[n] = state_out[n] + 0.5 * dt * k1[n];
        seirvd_rhs_c(tmp, k2, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                     vax_rate, mobility, daytime_pop);
        for (size_t n = 0; n < N; ++n) tmp[n] = state_out[n] + 0.5 * dt * k2[n];
        seirvd_rhs_c(tmp, k3, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                     vax_rate, mobility, daytime_pop);
        for (size_t n = 0; n < N; ++n) tmp[n] = state_out[n] + dt * k3[n];
        seirvd_rhs_c(tmp, k4, G, beta, sigma, gam, omega, VE, V_waning, ifr,
                     vax_rate, mobility, daytime_pop);
        const double s6 = dt / 6.0;
        for (size_t n = 0; n < N; ++n) {
            double v = state_out[n] + s6 * (k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n]);
            state_out[n] = v > 0.0 ? v : 0.0;
        }
    }

    free(k1); free(k2); free(k3); free(k4); free(tmp);
}

/* Exposed FoI (optional — Python may call rk4_step_c as a whole).
 * Kept for parity with the Rust implementation. */
SEIR_EXPORT void
commuter_foi_c(const double *mobility, const double *I_infectious,
               const double *population, int G, double beta, double *foi_out)
{
    double prev_stack[SEIR_STACK_G];
    double *prev = prev_stack;
    double *heap = NULL;
    if (G > SEIR_STACK_G) {
        heap = (double *)malloc((size_t)G * sizeof(double));
        if (!heap) { memset(foi_out, 0, (size_t)G * sizeof(double)); return; }
        prev = heap;
    }
    for (int j = 0; j < G; ++j) {
        double p = population[j];
        if (p < 1.0) p = 1.0;
        prev[j] = I_infectious[j] / p;
    }
    for (int i = 0; i < G; ++i) {
        double s = 0.0;
        const double *row = &mobility[i * G];
        for (int j = 0; j < G; ++j) s += row[j] * prev[j];
        foi_out[i] = beta * s;
    }
    if (heap) free(heap);
}
