/* @ts-self-types="./seir_wasm.d.ts" */

/**
 * 질환 파라미터
 */
export class DiseaseParams {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DiseaseParamsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_diseaseparams_free(ptr, 0);
    }
    /**
     * 전파율 β = R₀ × γ
     * @returns {number}
     */
    beta() {
        const ret = wasm.diseaseparams_beta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} r0
     * @param {number} incubation_days
     * @param {number} infectious_days
     * @param {number} cfr
     * @param {number} vax_rate
     * @param {number} vax_eff
     */
    constructor(r0, incubation_days, infectious_days, cfr, vax_rate, vax_eff) {
        const ret = wasm.diseaseparams_new(r0, incubation_days, infectious_days, cfr, vax_rate, vax_eff);
        this.__wbg_ptr = ret >>> 0;
        DiseaseParamsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get gamma() {
        const ret = wasm.__wbg_get_diseaseparams_gamma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get mu() {
        const ret = wasm.__wbg_get_diseaseparams_mu(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get r0() {
        const ret = wasm.__wbg_get_diseaseparams_r0(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get sigma() {
        const ret = wasm.__wbg_get_diseaseparams_sigma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get vax_eff() {
        const ret = wasm.__wbg_get_diseaseparams_vax_eff(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get vax_rate() {
        const ret = wasm.__wbg_get_diseaseparams_vax_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set gamma(arg0) {
        wasm.__wbg_set_diseaseparams_gamma(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set mu(arg0) {
        wasm.__wbg_set_diseaseparams_mu(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set r0(arg0) {
        wasm.__wbg_set_diseaseparams_r0(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set sigma(arg0) {
        wasm.__wbg_set_diseaseparams_sigma(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set vax_eff(arg0) {
        wasm.__wbg_set_diseaseparams_vax_eff(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set vax_rate(arg0) {
        wasm.__wbg_set_diseaseparams_vax_rate(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) DiseaseParams.prototype[Symbol.dispose] = DiseaseParams.prototype.free;

/**
 * 정책 개입 파라미터
 */
export class Intervention {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        InterventionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intervention_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get distancing_effect() {
        const ret = wasm.__wbg_get_intervention_distancing_effect(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get end_day() {
        const ret = wasm.__wbg_get_intervention_end_day(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get mask_effect() {
        const ret = wasm.__wbg_get_intervention_mask_effect(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get quarantine_rate() {
        const ret = wasm.__wbg_get_intervention_quarantine_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get school_close_effect() {
        const ret = wasm.__wbg_get_intervention_school_close_effect(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get start_day() {
        const ret = wasm.__wbg_get_intervention_start_day(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.intervention_new();
        this.__wbg_ptr = ret >>> 0;
        InterventionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} arg0
     */
    set distancing_effect(arg0) {
        wasm.__wbg_set_intervention_distancing_effect(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set end_day(arg0) {
        wasm.__wbg_set_intervention_end_day(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set mask_effect(arg0) {
        wasm.__wbg_set_intervention_mask_effect(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set quarantine_rate(arg0) {
        wasm.__wbg_set_intervention_quarantine_rate(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set school_close_effect(arg0) {
        wasm.__wbg_set_intervention_school_close_effect(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set start_day(arg0) {
        wasm.__wbg_set_intervention_start_day(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) Intervention.prototype[Symbol.dispose] = Intervention.prototype.free;

/**
 * 시뮬레이션 결과 — 시계열 데이터
 */
export class SimResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(SimResult.prototype);
        obj.__wbg_ptr = ptr;
        SimResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SimResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_simresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get_attack_rates() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_attack_rates(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} gu
     * @returns {Float64Array}
     */
    get_d(gu) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_d(retptr, this.__wbg_ptr, gu);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} gu
     * @returns {Float64Array}
     */
    get_daily_new(gu) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_daily_new(retptr, this.__wbg_ptr, gu);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} gu
     * @returns {Float64Array}
     */
    get_e(gu) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_e(retptr, this.__wbg_ptr, gu);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} gu
     * @returns {Float64Array}
     */
    get_i(gu) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_i(retptr, this.__wbg_ptr, gu);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} gu
     * @returns {Float64Array}
     */
    get_r(gu) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_r(retptr, this.__wbg_ptr, gu);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {Float64Array}
     */
    get_re_t() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_re_t(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * 특정 구의 시계열 반환 (JS Float64Array)
     * @param {number} gu
     * @returns {Float64Array}
     */
    get_s(gu) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_s(retptr, this.__wbg_ptr, gu);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {Float64Array}
     */
    get_total_i() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_total_i(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {Float64Array}
     */
    get_total_new() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_total_new(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * 전체 서울 합산 시계열
     * @returns {Float64Array}
     */
    get_total_s() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.simresult_get_total_s(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number}
     */
    n_days() {
        const ret = wasm.simresult_n_days(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    n_gu() {
        const ret = wasm.simresult_n_gu(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) SimResult.prototype[Symbol.dispose] = SimResult.prototype.free;

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
 * @param {Float64Array} populations
 * @param {number} initial_infected_gu
 * @param {number} initial_infected_n
 * @param {Float64Array} mobility_flat
 * @param {DiseaseParams} params
 * @param {Intervention} intervention
 * @param {number} n_days
 * @param {number} dt
 * @returns {SimResult}
 */
export function run_seir_metapop(populations, initial_infected_gu, initial_infected_n, mobility_flat, params, intervention, n_days, dt) {
    const ptr0 = passArrayF64ToWasm0(populations, wasm.__wbindgen_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(mobility_flat, wasm.__wbindgen_export);
    const len1 = WASM_VECTOR_LEN;
    _assertClass(params, DiseaseParams);
    _assertClass(intervention, Intervention);
    const ret = wasm.run_seir_metapop(ptr0, len0, initial_infected_gu, initial_infected_n, ptr1, len1, params.__wbg_ptr, intervention.__wbg_ptr, n_days, dt);
    return SimResult.__wrap(ret);
}

/**
 * 단일 구역 SEIR 시뮬레이션 (공간 없이, 빠른 테스트용)
 * @param {number} population
 * @param {number} initial_infected
 * @param {DiseaseParams} params
 * @param {Intervention} intervention
 * @param {number} n_days
 * @returns {SimResult}
 */
export function run_seir_single(population, initial_infected, params, intervention, n_days) {
    _assertClass(params, DiseaseParams);
    _assertClass(intervention, Intervention);
    const ret = wasm.run_seir_single(population, initial_infected, params.__wbg_ptr, intervention.__wbg_ptr, n_days);
    return SimResult.__wrap(ret);
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_6b64449b9b9ed33c: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
    };
    return {
        __proto__: null,
        "./seir_wasm_bg.js": import0,
    };
}

const DiseaseParamsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_diseaseparams_free(ptr >>> 0, 1));
const InterventionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intervention_free(ptr >>> 0, 1));
const SimResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_simresult_free(ptr >>> 0, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('seir_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
