/* tslint:disable */
/* eslint-disable */

/**
 * 질환 파라미터
 */
export class DiseaseParams {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * 전파율 β = R₀ × γ
     */
    beta(): number;
    constructor(r0: number, incubation_days: number, infectious_days: number, cfr: number, vax_rate: number, vax_eff: number);
    gamma: number;
    mu: number;
    r0: number;
    sigma: number;
    vax_eff: number;
    vax_rate: number;
}

/**
 * 정책 개입 파라미터
 */
export class Intervention {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    distancing_effect: number;
    end_day: number;
    mask_effect: number;
    quarantine_rate: number;
    school_close_effect: number;
    start_day: number;
}

/**
 * 시뮬레이션 결과 — 시계열 데이터
 */
export class SimResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get_attack_rates(): Float64Array;
    get_d(gu: number): Float64Array;
    get_daily_new(gu: number): Float64Array;
    get_e(gu: number): Float64Array;
    get_i(gu: number): Float64Array;
    get_r(gu: number): Float64Array;
    get_re_t(): Float64Array;
    /**
     * 특정 구의 시계열 반환 (JS Float64Array)
     */
    get_s(gu: number): Float64Array;
    get_total_i(): Float64Array;
    get_total_new(): Float64Array;
    /**
     * 전체 서울 합산 시계열
     */
    get_total_s(): Float64Array;
    n_days(): number;
    n_gu(): number;
}

/**
 * 메타인구 SEIR-V-D 시뮬레이션 실행
 *
 * # Arguments
 * * `populations` - 구별 인구 (Float64Array, length=n_gu)
 * * `initial_infected_gu` - 초기 감염 발생 자치구 인덱스
 * * `initial_infected_n` - 초기 감염자 수
 * * `mobility_flat` - 이동 행렬 (Float64Array, n_gu×n_gu, row-major)
 * * `params` - 질환 파라미터
 * * `intervention` - 정책 개입
 * * `n_days` - 시뮬레이션 일수
 * * `dt` - 시간 간격 (기본 0.25 = 6시간)
 */
export function run_seir_metapop(populations: Float64Array, initial_infected_gu: number, initial_infected_n: number, mobility_flat: Float64Array, params: DiseaseParams, intervention: Intervention, n_days: number, dt: number): SimResult;

/**
 * 단일 구역 SEIR 시뮬레이션 (공간 없이, 빠른 테스트용)
 */
export function run_seir_single(population: number, initial_infected: number, params: DiseaseParams, intervention: Intervention, n_days: number): SimResult;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_diseaseparams_free: (a: number, b: number) => void;
    readonly __wbg_get_diseaseparams_gamma: (a: number) => number;
    readonly __wbg_get_diseaseparams_mu: (a: number) => number;
    readonly __wbg_get_diseaseparams_r0: (a: number) => number;
    readonly __wbg_get_diseaseparams_sigma: (a: number) => number;
    readonly __wbg_get_diseaseparams_vax_eff: (a: number) => number;
    readonly __wbg_get_diseaseparams_vax_rate: (a: number) => number;
    readonly __wbg_get_intervention_end_day: (a: number) => number;
    readonly __wbg_get_intervention_start_day: (a: number) => number;
    readonly __wbg_intervention_free: (a: number, b: number) => void;
    readonly __wbg_set_diseaseparams_gamma: (a: number, b: number) => void;
    readonly __wbg_set_diseaseparams_mu: (a: number, b: number) => void;
    readonly __wbg_set_diseaseparams_r0: (a: number, b: number) => void;
    readonly __wbg_set_diseaseparams_sigma: (a: number, b: number) => void;
    readonly __wbg_set_diseaseparams_vax_eff: (a: number, b: number) => void;
    readonly __wbg_set_diseaseparams_vax_rate: (a: number, b: number) => void;
    readonly __wbg_set_intervention_end_day: (a: number, b: number) => void;
    readonly __wbg_set_intervention_start_day: (a: number, b: number) => void;
    readonly __wbg_simresult_free: (a: number, b: number) => void;
    readonly diseaseparams_beta: (a: number) => number;
    readonly diseaseparams_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly intervention_new: () => number;
    readonly run_seir_metapop: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
    readonly run_seir_single: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly simresult_get_attack_rates: (a: number, b: number) => void;
    readonly simresult_get_d: (a: number, b: number, c: number) => void;
    readonly simresult_get_daily_new: (a: number, b: number, c: number) => void;
    readonly simresult_get_e: (a: number, b: number, c: number) => void;
    readonly simresult_get_i: (a: number, b: number, c: number) => void;
    readonly simresult_get_r: (a: number, b: number, c: number) => void;
    readonly simresult_get_re_t: (a: number, b: number) => void;
    readonly simresult_get_s: (a: number, b: number, c: number) => void;
    readonly simresult_get_total_i: (a: number, b: number) => void;
    readonly simresult_get_total_new: (a: number, b: number) => void;
    readonly simresult_get_total_s: (a: number, b: number) => void;
    readonly simresult_n_days: (a: number) => number;
    readonly simresult_n_gu: (a: number) => number;
    readonly __wbg_get_intervention_distancing_effect: (a: number) => number;
    readonly __wbg_get_intervention_mask_effect: (a: number) => number;
    readonly __wbg_get_intervention_quarantine_rate: (a: number) => number;
    readonly __wbg_get_intervention_school_close_effect: (a: number) => number;
    readonly __wbg_set_intervention_distancing_effect: (a: number, b: number) => void;
    readonly __wbg_set_intervention_mask_effect: (a: number, b: number) => void;
    readonly __wbg_set_intervention_quarantine_rate: (a: number, b: number) => void;
    readonly __wbg_set_intervention_school_close_effect: (a: number, b: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
