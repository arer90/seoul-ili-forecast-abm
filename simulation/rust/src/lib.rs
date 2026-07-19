//! Rust-accelerated SEIR-V-D RK4 stepper + commuter-coupled FoI.
//!
//! Matches `simulation/sim/stepper.py::rk4_step_jit` semantics exactly
//! (deterministic RK4, 6 compartments S/E/I/R/V/D). Built as a Python
//! extension via PyO3 + maturin.
//!
//! Build:
//!   uv pip install maturin
//!   cd simulation/rust && maturin develop --release
//!
//! Python loader:
//!   from seir_core import rk4_step_rs, commuter_foi_rs
//!
//! ABI: all arrays float64 contiguous. Matches Numba/C stepper byte-for-byte
//! modulo floating-point non-associativity (values agree to ~1e-12).

use ndarray::{Array1, Array2, Axis};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const AGENT_STATE_S: i8 = 0;
const AGENT_STATE_E: i8 = 1;
const AGENT_STATE_I: i8 = 2;
const AGENT_STATE_R: i8 = 3;
const AGENT_STATE_V: i8 = 4;
const AGENT_STATE_D: i8 = 5;
const AGENT_N_COMPARTMENTS: usize = 6;
const AGENT_N_GU: usize = 25;
const AGENT_N_AGE: usize = 7;
const AGENT_DT: f64 = 1.0;
const AGENT_INITIAL_INFECTED_FRAC: f64 = 0.01;
const AGENT_RISK_DECAY: f64 = 1.0 / 30.0;
const AGENT_FATIGUE_ACCRUAL: f64 = 0.05;
const AGENT_COMPLIANCE_STRENGTH: f64 = 0.6;
const AGENT_COMPLIANCE_STEEPNESS: f64 = 30.0;
const AGENT_INIT_STREAM: u64 = 917_531;

/// SEIR-V-D right-hand side. Out-buffer filled in place.
#[inline(always)]
fn seirvd_rhs(
    state: &Array2<f64>,
    rhs: &mut Array2<f64>,
    beta: f64,
    sigma: f64,
    gamma: f64,
    omega: f64,
    ve: f64,
    v_waning: f64,
    ifr: f64,
    vax_rate: &Array1<f64>,
    mobility: &Array2<f64>,
    daytime_pop: &Array1<f64>,
) {
    let g = state.shape()[0];

    // I_present[j] = Σ_i mobility[i, j] * I[i]
    let mut i_present = Array1::<f64>::zeros(g);
    for j in 0..g {
        let mut s = 0.0;
        for i in 0..g {
            s += mobility[[i, j]] * state[[i, 2]];
        }
        i_present[j] = s;
    }

    // prev_j[j] = I_present[j] / max(daytime_pop[j], 1)
    let mut prev_j = Array1::<f64>::zeros(g);
    for j in 0..g {
        let d = daytime_pop[j].max(1.0);
        prev_j[j] = i_present[j] / d;
    }

    // lam[i] = beta * Σ_j mobility[i, j] * prev_j[j], clipped ≥ 0
    let mut lam = Array1::<f64>::zeros(g);
    for i in 0..g {
        let mut s = 0.0;
        for j in 0..g {
            s += mobility[[i, j]] * prev_j[j];
        }
        let v = beta * s;
        lam[i] = if v > 0.0 { v } else { 0.0 };
    }

    // RHS
    for i in 0..g {
        let s_i = state[[i, 0]];
        let e_i = state[[i, 1]];
        let i_i = state[[i, 2]];
        let r_i = state[[i, 3]];
        let v_i = state[[i, 4]];
        let lm = lam[i];
        let lmv = (1.0 - ve) * lm;
        let vr = vax_rate[i];

        rhs[[i, 0]] = -lm * s_i - vr * s_i + omega * r_i + v_waning * v_i;
        rhs[[i, 1]] = lm * s_i + lmv * v_i - sigma * e_i;
        rhs[[i, 2]] = sigma * e_i - gamma * i_i;
        rhs[[i, 3]] = gamma * (1.0 - ifr) * i_i - omega * r_i;
        rhs[[i, 4]] = vr * s_i - lmv * v_i - v_waning * v_i;
        rhs[[i, 5]] = gamma * ifr * i_i;
    }
}

/// Deterministic RK4 step for the metapop SEIR-V-D ODE.
///
/// Returns a fresh (G, 6) float64 array. Matches rk4_step_jit semantics.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rk4_step_rs<'py>(
    py: Python<'py>,
    state: PyReadonlyArray2<f64>,
    dt: f64,
    beta: f64,
    sigma: f64,
    gamma: f64,
    omega: f64,
    ve: f64,
    v_waning: f64,
    ifr: f64,
    vax_rate: PyReadonlyArray1<f64>,
    _populations: PyReadonlyArray1<f64>, // parity with Numba signature
    mobility: PyReadonlyArray2<f64>,
    daytime_pop: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let state_arr = state.as_array().to_owned();
    let vax_arr = vax_rate.as_array().to_owned();
    let mob_arr = mobility.as_array().to_owned();
    let day_arr = daytime_pop.as_array().to_owned();

    let g = state_arr.shape()[0];
    let mut k1 = Array2::<f64>::zeros((g, 6));
    let mut k2 = Array2::<f64>::zeros((g, 6));
    let mut k3 = Array2::<f64>::zeros((g, 6));
    let mut k4 = Array2::<f64>::zeros((g, 6));

    seirvd_rhs(&state_arr, &mut k1, beta, sigma, gamma, omega, ve, v_waning, ifr,
               &vax_arr, &mob_arr, &day_arr);
    let mut tmp = &state_arr + &(0.5 * dt * &k1);
    seirvd_rhs(&tmp, &mut k2, beta, sigma, gamma, omega, ve, v_waning, ifr,
               &vax_arr, &mob_arr, &day_arr);
    tmp = &state_arr + &(0.5 * dt * &k2);
    seirvd_rhs(&tmp, &mut k3, beta, sigma, gamma, omega, ve, v_waning, ifr,
               &vax_arr, &mob_arr, &day_arr);
    tmp = &state_arr + &(dt * &k3);
    seirvd_rhs(&tmp, &mut k4, beta, sigma, gamma, omega, ve, v_waning, ifr,
               &vax_arr, &mob_arr, &day_arr);

    let s6 = dt / 6.0;
    let mut out = &state_arr + &(s6 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4));

    // Clamp ≥ 0
    for v in out.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    Ok(out.into_pyarray_bound(py))
}

/// Batched N-step RK4 — runs `n_steps` sub-steps inside the Rust kernel so
/// the PyO3 boundary is crossed only once per batch. Shows the ceiling of
/// Rust performance when overhead is fully amortized.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rk4_step_batch_rs<'py>(
    py: Python<'py>,
    state: PyReadonlyArray2<f64>,
    n_steps: usize,
    dt: f64,
    beta: f64,
    sigma: f64,
    gamma: f64,
    omega: f64,
    ve: f64,
    v_waning: f64,
    ifr: f64,
    vax_rate: PyReadonlyArray1<f64>,
    _populations: PyReadonlyArray1<f64>,
    mobility: PyReadonlyArray2<f64>,
    daytime_pop: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut s = state.as_array().to_owned();
    let vax_arr = vax_rate.as_array().to_owned();
    let mob_arr = mobility.as_array().to_owned();
    let day_arr = daytime_pop.as_array().to_owned();

    let g = s.shape()[0];
    let mut k1 = Array2::<f64>::zeros((g, 6));
    let mut k2 = Array2::<f64>::zeros((g, 6));
    let mut k3 = Array2::<f64>::zeros((g, 6));
    let mut k4 = Array2::<f64>::zeros((g, 6));

    for _ in 0..n_steps {
        seirvd_rhs(&s, &mut k1, beta, sigma, gamma, omega, ve, v_waning, ifr,
                   &vax_arr, &mob_arr, &day_arr);
        let t1 = &s + &(0.5 * dt * &k1);
        seirvd_rhs(&t1, &mut k2, beta, sigma, gamma, omega, ve, v_waning, ifr,
                   &vax_arr, &mob_arr, &day_arr);
        let t2 = &s + &(0.5 * dt * &k2);
        seirvd_rhs(&t2, &mut k3, beta, sigma, gamma, omega, ve, v_waning, ifr,
                   &vax_arr, &mob_arr, &day_arr);
        let t3 = &s + &(dt * &k3);
        seirvd_rhs(&t3, &mut k4, beta, sigma, gamma, omega, ve, v_waning, ifr,
                   &vax_arr, &mob_arr, &day_arr);

        let s6 = dt / 6.0;
        s = &s + &(s6 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4));
        for v in s.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }

    Ok(s.into_pyarray_bound(py))
}

/// Commuter-coupled force of infection.
/// FoI_k = beta * Σ_j (commuter_matrix[k,j] * I[j] / N[j]).
#[pyfunction]
fn commuter_foi_rs<'py>(
    py: Python<'py>,
    commuter_matrix: PyReadonlyArray2<f64>,
    i_infectious: PyReadonlyArray1<f64>,
    population: PyReadonlyArray1<f64>,
    beta: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let c: Array2<f64> = commuter_matrix.as_array().to_owned();
    let i_arr: Array1<f64> = i_infectious.as_array().to_owned();
    let pop: Array1<f64> = population.as_array().to_owned();

    let g = c.shape()[0];
    let mut foi = Array1::<f64>::zeros(g);
    let prev: Array1<f64> = &i_arr / &pop.mapv(|p| p.max(1.0));

    for (k, row) in c.axis_iter(Axis(0)).enumerate() {
        foi[k] = beta * row.iter().zip(prev.iter()).map(|(cc, pp)| cc * pp).sum::<f64>();
    }

    Ok(foi.into_pyarray_bound(py))
}

fn py_value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn validate_rate(name: &str, value: f64) -> PyResult<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(py_value_error(format!(
            "{name} must be finite and >= 0; got {value:?}"
        )));
    }
    Ok(value)
}

fn validate_nonnegative_finite(name: &str, value: f64) -> PyResult<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(py_value_error(format!(
            "{name} must be finite and >= 0; got {value:?}"
        )));
    }
    Ok(value)
}

fn validate_tau(value: f64) -> PyResult<f64> {
    if value <= 0.0 || value.is_nan() {
        return Err(py_value_error(format!(
            "tau_mean must be positive or +inf; got {value:?}"
        )));
    }
    Ok(value)
}

fn any_to_f64_array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let np = py.import_bound("numpy")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    let arr_any = np.getattr("asarray")?.call((obj,), Some(&kwargs))?;
    let readonly = arr_any.extract::<PyReadonlyArrayDyn<f64>>()?;
    let arr = readonly.as_array();
    Ok((arr.iter().copied().collect(), arr.shape().to_vec()))
}

fn py_i64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let readonly = obj.extract::<PyReadonlyArrayDyn<i64>>()?;
    Ok(readonly.as_array().iter().copied().collect())
}

fn py_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let readonly = obj.extract::<PyReadonlyArrayDyn<f64>>()?;
    Ok(readonly.as_array().iter().copied().collect())
}

fn normalise_probability_row(row: &[f64], name: &str) -> PyResult<Vec<f64>> {
    if row.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(py_value_error(format!(
            "{name} must contain finite nonnegative probabilities"
        )));
    }
    let total: f64 = row.iter().sum();
    if total <= 0.0 {
        return Err(py_value_error(format!("{name} must have positive mass")));
    }
    Ok(row.iter().map(|v| *v / total).collect())
}

fn as_gu_rate<'py>(
    py: Python<'py>,
    name: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Vec<f64>> {
    let (data, shape) = any_to_f64_array(py, value)?;
    if shape.is_empty() {
        let rate = validate_rate(name, data[0])?;
        return Ok(vec![rate; AGENT_N_GU]);
    }
    if shape != [AGENT_N_GU] {
        return Err(py_value_error(format!(
            "{name} must be scalar or shape ({AGENT_N_GU},); got {:?}",
            shape
        )));
    }
    if data.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(py_value_error(format!(
            "{name} must contain finite nonnegative rates"
        )));
    }
    Ok(data)
}

fn normalise_mixing_matrix<'py>(
    py: Python<'py>,
    mixing_matrix: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Vec<f64>>> {
    let Some(obj) = mixing_matrix else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    let (data, shape) = any_to_f64_array(py, obj)?;
    if shape != [AGENT_N_GU, AGENT_N_GU] {
        return Err(py_value_error(format!(
            "mixing_matrix must have shape ({AGENT_N_GU}, {AGENT_N_GU}); got {:?}",
            shape
        )));
    }
    if data.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(py_value_error(
            "mixing_matrix must contain finite nonnegative values",
        ));
    }
    let mut out = vec![0.0; AGENT_N_GU * AGENT_N_GU];
    for gu in 0..AGENT_N_GU {
        let lo = gu * AGENT_N_GU;
        let row_sum: f64 = data[lo..lo + AGENT_N_GU].iter().sum();
        if row_sum <= 0.0 {
            return Err(py_value_error("mixing_matrix rows must have positive sums"));
        }
        for j in 0..AGENT_N_GU {
            out[lo + j] = data[lo + j] / row_sum;
        }
    }
    Ok(Some(out))
}

fn normalise_age_dist<'py>(
    py: Python<'py>,
    age_dist: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<f64>> {
    let Some(obj) = age_dist else {
        let probs = [0.06, 0.09, 0.12, 0.16, 0.18, 0.20, 0.19];
        let row = normalise_probability_row(&probs, "age_dist")?;
        return Ok((0..AGENT_N_GU).flat_map(|_| row.iter().copied()).collect());
    };
    if obj.is_none() {
        let probs = [0.06, 0.09, 0.12, 0.16, 0.18, 0.20, 0.19];
        let row = normalise_probability_row(&probs, "age_dist")?;
        return Ok((0..AGENT_N_GU).flat_map(|_| row.iter().copied()).collect());
    }
    let (data, shape) = any_to_f64_array(py, obj)?;
    if shape == [AGENT_N_AGE] {
        let row = normalise_probability_row(&data, "age_dist")?;
        return Ok((0..AGENT_N_GU).flat_map(|_| row.iter().copied()).collect());
    }
    if shape != [AGENT_N_GU, AGENT_N_AGE] {
        return Err(py_value_error(format!(
            "age_dist must be shape ({AGENT_N_AGE},) or ({AGENT_N_GU}, {AGENT_N_AGE}); got {:?}",
            shape
        )));
    }
    let mut out = vec![0.0; AGENT_N_GU * AGENT_N_AGE];
    for gu in 0..AGENT_N_GU {
        let lo = gu * AGENT_N_AGE;
        let row = normalise_probability_row(
            &data[lo..lo + AGENT_N_AGE],
            &format!("age_dist[{gu}]"),
        )?;
        out[lo..lo + AGENT_N_AGE].copy_from_slice(&row);
    }
    Ok(out)
}

fn build_home_gu(n: usize) -> (Vec<i8>, Vec<usize>) {
    let mut counts = vec![n / AGENT_N_GU; AGENT_N_GU];
    for c in counts.iter_mut().take(n % AGENT_N_GU) {
        *c += 1;
    }
    let mut offsets = Vec::with_capacity(AGENT_N_GU + 1);
    offsets.push(0);
    for c in &counts {
        offsets.push(offsets.last().copied().unwrap() + *c);
    }
    let mut home_gu = Vec::with_capacity(n);
    for (gu, c) in counts.iter().enumerate() {
        home_gu.extend(std::iter::repeat(gu as i8).take(*c));
    }
    (home_gu, offsets)
}

fn hazard(rate: f64) -> f64 {
    1.0 - (-rate * AGENT_DT).exp()
}

fn logistic(x: f64) -> f64 {
    let z = x.clamp(-60.0, 60.0);
    1.0 / (1.0 + (-z).exp())
}

fn record_agent_counts(state: &[i8], out: &mut [Vec<i64>], day: usize) {
    let mut counts = [0_i64; AGENT_N_COMPARTMENTS];
    for s in state {
        let idx = *s as usize;
        if idx < AGENT_N_COMPARTMENTS {
            counts[idx] += 1;
        }
    }
    for idx in 0..AGENT_N_COMPARTMENTS {
        out[idx][day] = counts[idx];
    }
}

struct GuRandom {
    s: Vec<f64>,
    e: Vec<f64>,
    i: Vec<f64>,
}

struct GuUpdate {
    gu: usize,
    next_state: Vec<i8>,
    fatigue: Vec<f64>,
    compliance: Vec<f64>,
}

/// Rust/PyO3 port of `simulation.abm.agent_kernel.run_agent_world`.
///
/// The public Python signature intentionally mirrors the NumPy oracle. Random
/// streams are materialized per `(global_seed, day, gu)` before Rayon updates
/// the 25 gu slices, so the trajectory is independent of Rayon scheduling.
#[pyfunction]
#[allow(clippy::too_many_arguments, non_snake_case)]
#[pyo3(signature = (
    N,
    T_days,
    beta,
    sigma,
    gamma,
    delta,
    nu,
    mixing_matrix=None,
    age_dist=None,
    global_seed=42,
    theta_mean=0.5,
    theta_sd=0.15,
    alpha_mean=0.3,
    kappa_mean=0.5,
    tau_mean=7.0
))]
pub fn run_agent_world_rs<'py>(
    py: Python<'py>,
    N: usize,
    T_days: usize,
    beta: f64,
    sigma: f64,
    gamma: f64,
    delta: f64,
    nu: &Bound<'py, PyAny>,
    mixing_matrix: Option<&Bound<'py, PyAny>>,
    age_dist: Option<&Bound<'py, PyAny>>,
    global_seed: u64,
    theta_mean: f64,
    theta_sd: f64,
    alpha_mean: f64,
    kappa_mean: f64,
    tau_mean: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if N < 1 {
        return Err(py_value_error(format!("N must be >= 1; got {N}")));
    }
    if T_days < 1 {
        return Err(py_value_error(format!(
            "T_days must be >= 1; got {T_days}"
        )));
    }
    let beta = validate_rate("beta", beta)?;
    let sigma = validate_rate("sigma", sigma)?;
    let gamma = validate_rate("gamma", gamma)?;
    let delta = validate_rate("delta", delta)?;
    let nu_by_gu = as_gu_rate(py, "nu", nu)?;
    let theta_mean = validate_nonnegative_finite("theta_mean", theta_mean)?;
    let theta_sd = validate_nonnegative_finite("theta_sd", theta_sd)?;
    let alpha_mean = validate_nonnegative_finite("alpha_mean", alpha_mean)?;
    let kappa_mean = validate_nonnegative_finite("kappa_mean", kappa_mean)?;
    let tau_mean = validate_tau(tau_mean)?;
    let mixing = normalise_mixing_matrix(py, mixing_matrix)?;
    let age_prob = normalise_age_dist(py, age_dist)?;

    let np_random = py.import_bound("numpy")?.getattr("random")?;
    let seed_sequence = np_random.getattr("SeedSequence")?;
    let default_rng = np_random.getattr("default_rng")?;

    let init_ss = seed_sequence.call1((vec![global_seed, AGENT_INIT_STREAM],))?;
    let init_rng = default_rng.call1((init_ss,))?;

    let (home_gu, gu_offsets) = build_home_gu(N);
    let mut work_gu = home_gu.clone();

    let mut age_band = vec![0_i8; N];
    for gu in 0..AGENT_N_GU {
        let lo = gu_offsets[gu];
        let hi = gu_offsets[gu + 1];
        let n_gu = hi - lo;
        if n_gu == 0 {
            continue;
        }
        let p = age_prob[gu * AGENT_N_AGE..(gu + 1) * AGENT_N_AGE]
            .to_vec()
            .into_pyarray_bound(py);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("size", n_gu)?;
        kwargs.set_item("p", p)?;
        let draw = init_rng.call_method("choice", (AGENT_N_AGE,), Some(&kwargs))?;
        let values = py_i64_vec(&draw)?;
        for (dst, src) in age_band[lo..hi].iter_mut().zip(values.iter()) {
            *dst = *src as i8;
        }
    }

    let mut state = vec![AGENT_STATE_S; N];
    let initial_i = (N as f64 * AGENT_INITIAL_INFECTED_FRAC).round() as usize;
    let initial_i = initial_i.max(1);
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("size", initial_i)?;
    kwargs.set_item("replace", false)?;
    let infected = init_rng.call_method("choice", (N,), Some(&kwargs))?;
    for idx in py_i64_vec(&infected)? {
        state[idx as usize] = AGENT_STATE_I;
    }

    let theta = if theta_sd == 0.0 {
        vec![theta_mean; N]
    } else {
        let normal = init_rng.call_method1("standard_normal", (N,))?;
        py_f64_vec(&normal)?
            .into_iter()
            .map(|z| (theta_mean * (1.0 + theta_sd * z)).max(0.0))
            .collect()
    };
    let alpha = vec![alpha_mean; N];
    let kappa = vec![kappa_mean; N];
    let tau = vec![tau_mean; N];
    let rho_value = if tau_mean.is_finite() {
        1.0 / tau_mean
    } else {
        0.0
    };
    let rho = vec![rho_value; N];
    let mut fatigue = vec![0.0; N];
    let mut compliance = vec![0.0; N];
    let mut risk_by_gu = vec![0.0; AGENT_N_GU];
    let mut alpha_by_gu = vec![0.0; AGENT_N_GU];
    for gu in 0..AGENT_N_GU {
        if gu_offsets[gu + 1] > gu_offsets[gu] {
            alpha_by_gu[gu] = alpha_mean;
        }
    }

    let mut out = (0..AGENT_N_COMPARTMENTS)
        .map(|_| vec![0_i64; T_days])
        .collect::<Vec<_>>();
    record_agent_counts(&state, &mut out, 0);

    let child_count = (T_days * AGENT_N_GU).max(1);
    let child_ss = seed_sequence.call1((global_seed,))?;
    let children = child_ss.call_method1("spawn", (child_count,))?;

    for out_day in 1..T_days {
        let day = out_day - 1;
        let mut rng_by_gu = Vec::with_capacity(AGENT_N_GU);
        for gu in 0..AGENT_N_GU {
            let child = children.get_item(day * AGENT_N_GU + gu)?;
            rng_by_gu.push(default_rng.call1((child,))?);
        }

        if let Some(mixing) = &mixing {
            for gu in 0..AGENT_N_GU {
                let lo = gu_offsets[gu];
                let hi = gu_offsets[gu + 1];
                let n_gu = hi - lo;
                if n_gu == 0 {
                    continue;
                }
                let p = mixing[gu * AGENT_N_GU..(gu + 1) * AGENT_N_GU]
                    .to_vec()
                    .into_pyarray_bound(py);
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("size", n_gu)?;
                kwargs.set_item("p", p)?;
                let draw = rng_by_gu[gu].call_method("choice", (AGENT_N_GU,), Some(&kwargs))?;
                let values = py_i64_vec(&draw)?;
                for (dst, src) in work_gu[lo..hi].iter_mut().zip(values.iter()) {
                    *dst = *src as i8;
                }
            }
        }

        let mut alive_total = 0_i64;
        let mut infected_total = 0_i64;
        let mut present_n = vec![0.0_f64; AGENT_N_GU];
        let mut present_i = vec![0.0_f64; AGENT_N_GU];
        let mut home_n = vec![0_i64; AGENT_N_GU];
        let mut home_i = vec![0_i64; AGENT_N_GU];
        for a in 0..N {
            if state[a] == AGENT_STATE_D {
                continue;
            }
            alive_total += 1;
            let home = home_gu[a] as usize;
            home_n[home] += 1;
            if state[a] == AGENT_STATE_I {
                infected_total += 1;
                home_i[home] += 1;
            }
            if mixing.is_some() {
                let work = work_gu[a] as usize;
                present_n[work] += 1.0;
                if state[a] == AGENT_STATE_I {
                    present_i[work] += 1.0;
                }
            }
        }

        let (global_prevalence, prevalence_by_work) = if mixing.is_some() {
            let prevalence = (0..AGENT_N_GU)
                .map(|gu| {
                    if present_n[gu] > 0.0 {
                        present_i[gu] / present_n[gu].max(1.0)
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            (0.0, Some(prevalence))
        } else {
            let prev = if alive_total > 0 {
                infected_total as f64 / alive_total as f64
            } else {
                0.0
            };
            (prev, None)
        };

        for gu in 0..AGENT_N_GU {
            let home_prev = if home_n[gu] > 0 {
                home_i[gu] as f64 / (home_n[gu] as f64).max(1.0)
            } else {
                0.0
            };
            risk_by_gu[gu] += alpha_by_gu[gu] * home_prev - AGENT_RISK_DECAY * risk_by_gu[gu];
            if risk_by_gu[gu] < 0.0 {
                risk_by_gu[gu] = 0.0;
            }
        }

        let mut random_by_gu = Vec::with_capacity(AGENT_N_GU);
        for gu in 0..AGENT_N_GU {
            let lo = gu_offsets[gu];
            let hi = gu_offsets[gu + 1];
            let current = &state[lo..hi];
            let s_count = current.iter().filter(|s| **s == AGENT_STATE_S).count();
            let e_count = current.iter().filter(|s| **s == AGENT_STATE_E).count();
            let i_count = current.iter().filter(|s| **s == AGENT_STATE_I).count();
            let rng = &rng_by_gu[gu];
            let s = if s_count > 0 {
                py_f64_vec(&rng.call_method1("random", (s_count,))?)?
            } else {
                Vec::new()
            };
            let e = if e_count > 0 {
                py_f64_vec(&rng.call_method1("random", (e_count,))?)?
            } else {
                Vec::new()
            };
            let i = if i_count > 0 {
                py_f64_vec(&rng.call_method1("random", (i_count,))?)?
            } else {
                Vec::new()
            };
            random_by_gu.push(GuRandom { s, e, i });
        }

        let updates = (0..AGENT_N_GU)
            .into_par_iter()
            .map(|gu| {
                let lo = gu_offsets[gu];
                let hi = gu_offsets[gu + 1];
                let current = &state[lo..hi];
                let mut next_state = current.to_vec();
                let mut new_compliance = vec![0.0; current.len()];
                let mut lam = vec![0.0; current.len()];

                for local in 0..current.len() {
                    let a = lo + local;
                    let margin = risk_by_gu[gu] - kappa[a] * fatigue[a] - theta[a];
                    let c = logistic(AGENT_COMPLIANCE_STEEPNESS * margin);
                    new_compliance[local] = c;
                    let contact_multiplier = 1.0 - AGENT_COMPLIANCE_STRENGTH * c;
                    lam[local] = if let Some(prevalence) = &prevalence_by_work {
                        beta * contact_multiplier * prevalence[work_gu[a] as usize]
                    } else {
                        beta * contact_multiplier * global_prevalence
                    };
                }

                let rng = &random_by_gu[gu];
                let mut s_rng_idx = 0;
                for (local, s) in current.iter().enumerate() {
                    if *s != AGENT_STATE_S {
                        continue;
                    }
                    let inf_rate = lam[local];
                    let vax_rate = nu_by_gu[gu];
                    let total_rate = inf_rate + vax_rate;
                    let (p_inf, p_vax) = if total_rate > 0.0 {
                        let p_out = hazard(total_rate);
                        let p_inf = p_out * inf_rate / total_rate;
                        (p_inf, p_out - p_inf)
                    } else {
                        (0.0, 0.0)
                    };
                    let u = rng.s[s_rng_idx];
                    s_rng_idx += 1;
                    if u < p_inf {
                        next_state[local] = AGENT_STATE_E;
                    } else if u < p_inf + p_vax {
                        next_state[local] = AGENT_STATE_V;
                    }
                }

                let p_ei = hazard(sigma);
                let mut e_rng_idx = 0;
                for (local, s) in current.iter().enumerate() {
                    if *s != AGENT_STATE_E {
                        continue;
                    }
                    let u = rng.e[e_rng_idx];
                    e_rng_idx += 1;
                    if u < p_ei {
                        next_state[local] = AGENT_STATE_I;
                    }
                }

                let total_i_rate = gamma + delta;
                let (p_rec, p_die) = if total_i_rate > 0.0 {
                    let p_out = hazard(total_i_rate);
                    let p_rec = p_out * gamma / total_i_rate;
                    (p_rec, p_out - p_rec)
                } else {
                    (0.0, 0.0)
                };
                let mut i_rng_idx = 0;
                for (local, s) in current.iter().enumerate() {
                    if *s != AGENT_STATE_I {
                        continue;
                    }
                    let u = rng.i[i_rng_idx];
                    i_rng_idx += 1;
                    if u < p_rec {
                        next_state[local] = AGENT_STATE_R;
                    } else if u < p_rec + p_die {
                        next_state[local] = AGENT_STATE_D;
                    }
                }

                let mut new_fatigue = vec![0.0; current.len()];
                for local in 0..current.len() {
                    let a = lo + local;
                    new_fatigue[local] = (fatigue[a]
                        + AGENT_FATIGUE_ACCRUAL * compliance[a]
                        - rho[a] * fatigue[a])
                        .max(0.0);
                }

                GuUpdate {
                    gu,
                    next_state,
                    fatigue: new_fatigue,
                    compliance: new_compliance,
                }
            })
            .collect::<Vec<_>>();

        for update in updates {
            let lo = gu_offsets[update.gu];
            let hi = gu_offsets[update.gu + 1];
            state[lo..hi].copy_from_slice(&update.next_state);
            fatigue[lo..hi].copy_from_slice(&update.fatigue);
            compliance[lo..hi].copy_from_slice(&update.compliance);
        }

        record_agent_counts(&state, &mut out, out_day);
    }

    let result = PyDict::new_bound(py);
    for (idx, name) in ["S", "E", "I", "R", "V", "D"].iter().enumerate() {
        result.set_item(*name, out[idx].clone().into_pyarray_bound(py))?;
    }
    let agents = PyDict::new_bound(py);
    agents.set_item("state", state.into_pyarray_bound(py))?;
    agents.set_item("age_band", age_band.into_pyarray_bound(py))?;
    agents.set_item("home_gu", home_gu.into_pyarray_bound(py))?;
    agents.set_item("work_gu", work_gu.into_pyarray_bound(py))?;
    agents.set_item("alpha", alpha.into_pyarray_bound(py))?;
    agents.set_item("kappa", kappa.into_pyarray_bound(py))?;
    agents.set_item("tau", tau.into_pyarray_bound(py))?;
    agents.set_item("theta", theta.into_pyarray_bound(py))?;
    agents.set_item("fatigue", fatigue.into_pyarray_bound(py))?;
    agents.set_item("compliance", compliance.into_pyarray_bound(py))?;
    result.set_item("agents", agents)?;
    Ok(result)
}

#[pymodule]
fn seir_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rk4_step_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_step_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(commuter_foi_rs, m)?)?;
    m.add_function(wrap_pyfunction!(run_agent_world_rs, m)?)?;
    Ok(())
}
