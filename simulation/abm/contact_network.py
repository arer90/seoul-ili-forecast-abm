"""명시적 다층 접촉망 (multi-layer contact network) — ABM 이질성 enrichment.

현재 ``contact_structure`` 는 연령×직업 평균장(WAIFW mean-field) 가정 — 모든
에이전트가 자기 구(區)의 평균 prevalence 에 노출된다. 본 모듈은 그 평균장 대신
**명시적 edge 기반** 다층 접촉망을 세운다: 같은 가구·직장·학교에 속한 에이전트만
서로 접촉한다. 이는 인플루엔자 전파의 지배적 증폭 채널(가구 2차감염, 학교 군집;
Cauchemez 2008, Mossong POLYMOD 2008)을 평균장이 평탄화해 버리는 문제를 해결한다.

네 개 layer:
  - **household** (가구): ``home_gu`` 내 작은 소그룹(크기 2-4). 가구 내 완전 그래프
    → degree ≈ hh_size-1. 인플루엔자 2차감염의 핵심.
  - **workplace** (직장): ``work_gu`` 내 ``occupation``(산업코드)별 직장 그룹.
    통근지에서 같은 산업끼리 접촉.
  - **school** (학교): 학령(age_band==1, "10-19") 에이전트가 ``home_gu`` 내 학교
    군집을 형성. ``affiliation._STUDENT_BAND`` 규약과 동일.
  - **community** (지역사회): ``home_gu`` 내 무작위 약결합(weak ties). 가구/직장/학교
    밖의 우발적 접촉 — 평균장 잔차를 대체.

**가법(additive) 설계**: ``agent_kernel``/``synthetic_population`` 을 import 만 하고
재사용한다(편집 X). 코어 평균장 kernel 은 그대로 두고, 본 모듈은 별도의 edge 기반
FoI(force of infection)를 제공해 평균장과의 이질성을 실측·비교할 수 있게 한다.

설계 철학(D-4 deep module): 작은 인터페이스 3함수(``build_multilayer_network`` ·
``network_foi`` · ``degree_summary``) + 풍부한 구현(층별 군집화·CSR 대칭 보장·층별
FoI 합산). 호출자는 ``population`` SoA dict 와 ``state`` 벡터만 알면 된다.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

# agent_kernel 의 compartment 코드(STATE_I 등)를 재사용 — import 만, 편집 X.
from simulation.abm.agent_kernel import STATE_D, STATE_I

__all__ = ["build_multilayer_network", "network_foi", "sample_infector",
           "degree_summary"]

# affiliation._STUDENT_BAND 와 동일한 학령 규약(age_band 1 = "10-19").
_STUDENT_BAND = 1
# 워크플레이스 layer 에서 미취업/무직(occupation 음수 코드)은 직장 edge 미형성.
_NO_WORKPLACE_CODE = -1
_LAYER_NAMES = ("household", "workplace", "school", "community")


def _clique_edges(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """한 그룹(완전 그래프)의 무방향 edge 를 (row, col) 상삼각 인덱스로 반환.

    Args:
        members: 그룹 구성원의 전역 에이전트 인덱스 1D 배열. 길이 < 2 면 edge 없음.

    Returns:
        ``(rows, cols)`` 정수 배열 쌍. 각 i<j 쌍을 한 번씩(상삼각)만 담는다 —
        대칭화는 호출자가 ``A + A.T`` 로 수행한다.

    Performance: O(k^2) (k=그룹 크기). 본 모듈은 가구·학교·직장의 군집 크기를
        제한(가구≤4, 직장/학교/community 는 청크)하므로 전체적으로 O(N)에 가깝다.
    Side effects: 없음.
    """
    k = members.shape[0]
    if k < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    iu, ju = np.triu_indices(k, k=1)
    return members[iu].astype(np.int64), members[ju].astype(np.int64)


def _chunk_groups(
    member_idx: np.ndarray, rng: np.random.Generator, *, low: int, high: int
) -> list[np.ndarray]:
    """주어진 에이전트 묶음을 [low, high] 크기의 무작위 소그룹들로 쪼갠다.

    구성원 순서를 섞은 뒤(결정성: 전달된 rng) [low, high] 균일분포에서 뽑은 크기로
    순차 분할한다. 마지막 잔여가 ``low`` 미만이면 직전 그룹에 흡수해 고립 노드(또는
    너무 작은 그룹)를 줄인다.

    Args:
        member_idx: 묶을 전역 에이전트 인덱스 1D 배열.
        rng: 결정적 셔플/크기 추출용 ``numpy`` Generator.
        low: 그룹 최소 크기(>=1).
        high: 그룹 최대 크기(>=low).

    Returns:
        각 원소가 한 소그룹의 인덱스 배열인 list. 모든 인덱스를 정확히 한 번 포함.

    Performance: O(k) (k=구성원 수).
    Side effects: ``rng`` 상태를 전진시킨다.
    """
    if member_idx.shape[0] == 0:
        return []
    shuffled = member_idx.copy()
    rng.shuffle(shuffled)
    groups: list[np.ndarray] = []
    pos = 0
    n = shuffled.shape[0]
    while pos < n:
        size = int(rng.integers(low, high + 1))
        size = max(1, min(size, n - pos))
        groups.append(shuffled[pos : pos + size])
        pos += size
    # 마지막 그룹이 너무 작으면(고립/2 미만) 직전 그룹에 흡수.
    if len(groups) >= 2 and groups[-1].shape[0] < low:
        groups[-2] = np.concatenate([groups[-2], groups[-1]])
        groups.pop()
    return groups


def _symmetric_csr(
    rows: np.ndarray, cols: np.ndarray, n: int
) -> sparse.csr_matrix:
    """상삼각 edge 목록을 무방향 대칭 0/1 CSR 인접행렬로 만든다.

    상삼각만 받은 뒤 ``A + A.T`` 로 대칭화하고, 중복/자기루프를 제거해 binary 화한다.

    Args:
        rows: edge 의 행 인덱스(i<j).
        cols: edge 의 열 인덱스.
        n: 노드 수(행렬은 n×n).

    Returns:
        대칭(``(A != A.T).nnz == 0``) 이고 대각이 0(자기루프 없음)이며 값이 모두 1인
        ``scipy.sparse.csr_matrix``.

    Performance: O(nnz) time, O(nnz) memory.
    Side effects: 없음.
    """
    if rows.shape[0] == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    data = np.ones(rows.shape[0], dtype=np.float64)
    upper = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    upper.setdiag(0.0)  # 자기루프 제거(같은 인덱스 우발 중복 방지)
    upper.eliminate_zeros()
    sym = upper + upper.T
    sym.data[:] = 1.0  # 중복 edge → binary
    sym = sym.tocsr()
    sym.eliminate_zeros()
    return sym


def _household_layer(
    home_gu: np.ndarray, rng: np.random.Generator, *, hh_size: tuple[int, int]
) -> sparse.csr_matrix:
    """가구 layer: 같은 ``home_gu`` 내 크기 [hh_size] 소그룹의 완전그래프."""
    n = home_gu.shape[0]
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    for gu in np.unique(home_gu):
        members = np.flatnonzero(home_gu == gu)
        for grp in _chunk_groups(members, rng, low=hh_size[0], high=hh_size[1]):
            r, c = _clique_edges(grp)
            if r.shape[0]:
                rows_all.append(r)
                cols_all.append(c)
    if not rows_all:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    return _symmetric_csr(
        np.concatenate(rows_all), np.concatenate(cols_all), n
    )


def _workplace_layer(
    work_gu: np.ndarray,
    occupation: np.ndarray,
    age_band: np.ndarray,
    rng: np.random.Generator,
    *,
    work_size: tuple[int, int],
) -> sparse.csr_matrix:
    """직장 layer: ``work_gu``×``occupation`` 별 성인 에이전트를 직장 소그룹으로.

    학령(age_band==1)·미취업(occupation<0)은 직장 edge 미형성. 같은 통근지·같은
    산업끼리 [work_size] 크기 소그룹의 완전그래프.
    """
    n = work_gu.shape[0]
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    employed = (occupation >= 0) & (age_band != _STUDENT_BAND)
    if not employed.any():
        return sparse.csr_matrix((n, n), dtype=np.float64)
    # (work_gu, occupation) 복합키로 그룹화 — 정렬해 결정적 순회.
    keys = work_gu.astype(np.int64) * (int(occupation.max()) + 2) + occupation.astype(
        np.int64
    )
    keys = np.where(employed, keys, -1)
    for key in np.unique(keys):
        if key < 0:
            continue
        members = np.flatnonzero(keys == key)
        for grp in _chunk_groups(members, rng, low=work_size[0], high=work_size[1]):
            r, c = _clique_edges(grp)
            if r.shape[0]:
                rows_all.append(r)
                cols_all.append(c)
    if not rows_all:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    return _symmetric_csr(
        np.concatenate(rows_all), np.concatenate(cols_all), n
    )


def _school_layer(
    home_gu: np.ndarray,
    age_band: np.ndarray,
    rng: np.random.Generator,
    *,
    class_size: tuple[int, int],
) -> sparse.csr_matrix:
    """학교 layer: 학령(age_band==1) 에이전트를 ``home_gu`` 내 학급 군집으로.

    ``affiliation`` 의 학교-군집 개념을 edge 형태로 구체화. 같은 구 학생들을
    [class_size] 크기 학급으로 분할한 완전그래프.
    """
    n = home_gu.shape[0]
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    students = age_band == _STUDENT_BAND
    if not students.any():
        return sparse.csr_matrix((n, n), dtype=np.float64)
    for gu in np.unique(home_gu[students]):
        members = np.flatnonzero(students & (home_gu == gu))
        for grp in _chunk_groups(
            members, rng, low=class_size[0], high=class_size[1]
        ):
            r, c = _clique_edges(grp)
            if r.shape[0]:
                rows_all.append(r)
                cols_all.append(c)
    if not rows_all:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    return _symmetric_csr(
        np.concatenate(rows_all), np.concatenate(cols_all), n
    )


def _community_layer(
    home_gu: np.ndarray, rng: np.random.Generator, *, mean_degree
) -> sparse.csr_matrix:
    """지역사회 layer: 같은 ``home_gu`` 내 무작위 약결합(Erdős–Rényi 근사).

    구별로 기대 평균 degree ``mean_degree`` 가 되도록 무작위 i<j 쌍을 표집한다.
    구 내 가능한 쌍 수 C(m,2) 에서 ``m*mean_degree/2`` 개를 비복원 추출.
    ``mean_degree`` 는 스칼라(전 구 동일) 또는 gu 코드로 인덱싱된 per-gu 배열
    (밀도/이동에서 데이터-유도) 둘 다 허용.
    """
    n = home_gu.shape[0]
    per_gu = np.ndim(mean_degree) > 0
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    for gu in np.unique(home_gu):
        members = np.flatnonzero(home_gu == gu)
        m = members.shape[0]
        if m < 2:
            continue
        total_pairs = m * (m - 1) // 2
        deg_g = float(mean_degree[int(gu)]) if per_gu else float(mean_degree)
        n_edges = int(round(m * deg_g / 2.0))
        n_edges = max(0, min(n_edges, total_pairs))
        if n_edges == 0:
            continue
        # 상삼각 선형 인덱스에서 비복원 추출 → (i, j) 복원.
        chosen = rng.choice(total_pairs, size=n_edges, replace=False)
        iu, ju = _linear_to_triu(chosen, m)
        rows_all.append(members[iu].astype(np.int64))
        cols_all.append(members[ju].astype(np.int64))
    if not rows_all:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    return _symmetric_csr(
        np.concatenate(rows_all), np.concatenate(cols_all), n
    )


def _linear_to_triu(lin: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """상삼각(i<j) 선형 인덱스를 (i, j) 좌표로 복원(community 표집용).

    행 i 의 시작 오프셋 base[i] = i*m - i*(i+1)/2 을 이용해 i 를 검색한 뒤 j 복원.
    """
    base = np.arange(m, dtype=np.int64) * m - np.arange(m, dtype=np.int64) * (
        np.arange(m, dtype=np.int64) + 1
    ) // 2
    i = np.searchsorted(base, lin, side="right") - 1
    j = lin - base[i] + i + 1
    return i.astype(np.int64), j.astype(np.int64)


def build_multilayer_network(
    population: dict[str, np.ndarray],
    *,
    seed: int,
    hh_size: tuple[int, int] = (2, 4),
    work_size: tuple[int, int] = (5, 20),
    class_size: tuple[int, int] = (20, 35),
    community_mean_degree: float = 4.0,
) -> dict[str, sparse.csr_matrix]:
    """서울 25구 합성 인구로부터 명시적 다층 접촉망을 구축한다.

    ``synthetic_population.generate_population`` 의 SoA dict 를 받아 네 layer
    (household/workplace/school/community)의 무방향 0/1 인접행렬을 만든다. 각
    layer 는 평균장과 달리 **누가 누구와 접촉하는지**를 명시한다. 결정성은 단일
    ``seed`` 로 보장(layer 별 child stream).

    Args:
        population: SoA dict. 최소한 ``home_gu``(0-24), ``work_gu``(0-24),
            ``age_band``(0-6, 1=학령), ``occupation``(>=0 코드, 음수=직장없음)
            키의 길이 N 1D 정수 배열을 포함해야 한다. ``generate_population`` 의
            반환 형식과 호환.
        seed: ``numpy.random.default_rng`` 시드(가구/직장/학급/약결합 추출).
        hh_size: 가구 그룹 크기 (low, high) 범위. degree ≈ 그룹크기-1.
        work_size: 직장 그룹 크기 (low, high) 범위.
        class_size: 학급 크기 (low, high) 범위.
        community_mean_degree: 구 내 약결합 layer 의 1인당 기대 degree.

    Returns:
        ``{"household", "workplace", "school", "community"}`` → N×N
        ``scipy.sparse.csr_matrix``. 각 행렬은 무방향 대칭(``(A!=A.T).nnz==0``),
        대각 0(자기루프 없음), 값 1.

    Raises:
        ValueError: ``population`` 에 필수 키가 없거나, 배열 길이가 불일치하거나,
            크기 범위가 ``low>high`` 또는 ``low<1`` 인 경우.

    Performance: O(N + E) time/memory (E=총 edge). 가구 O(N), 학교/직장은 청크
        완전그래프라 그룹크기에 선형, community 는 ``N*degree/2`` edge.
    Side effects: 없음(순수 함수). DB·디스크 접근 없음 — population 만 소비.
    Caller responsibility: population 의 정수 코드 범위(home_gu∈[0,24] 등)는
        ``generate_population`` 이 보장. 다른 출처면 호출자가 검증.
    """
    for key in ("home_gu", "work_gu", "age_band", "occupation"):
        if key not in population:
            raise ValueError(f"population missing required key {key!r}")
    home_gu = np.asarray(population["home_gu"], dtype=np.int64)
    work_gu = np.asarray(population["work_gu"], dtype=np.int64)
    age_band = np.asarray(population["age_band"], dtype=np.int64)
    occupation = np.asarray(population["occupation"], dtype=np.int64)
    n = home_gu.shape[0]
    for key, arr in (
        ("work_gu", work_gu),
        ("age_band", age_band),
        ("occupation", occupation),
    ):
        if arr.shape != (n,):
            raise ValueError(
                f"population[{key!r}] must have shape ({n},); got {arr.shape}"
            )
    for label, rng_size in (
        ("hh_size", hh_size),
        ("work_size", work_size),
        ("class_size", class_size),
    ):
        lo, hi = int(rng_size[0]), int(rng_size[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"{label} must satisfy 1<=low<=high; got {rng_size}")
    _cmd = np.asarray(community_mean_degree, dtype=float)
    if not np.all(np.isfinite(_cmd)) or np.any(_cmd < 0.0):
        raise ValueError(
            "community_mean_degree must be finite and >= 0 (scalar or per-gu array)")

    # layer 별 독립 child stream — 단일 seed 로 전체 결정성.
    streams = np.random.SeedSequence(int(seed)).spawn(4)
    rng_hh = np.random.default_rng(streams[0])
    rng_work = np.random.default_rng(streams[1])
    rng_school = np.random.default_rng(streams[2])
    rng_comm = np.random.default_rng(streams[3])

    layers = {
        "household": _household_layer(home_gu, rng_hh, hh_size=hh_size),
        "workplace": _workplace_layer(
            work_gu, occupation, age_band, rng_work, work_size=work_size
        ),
        "school": _school_layer(
            home_gu, age_band, rng_school, class_size=class_size
        ),
        "community": _community_layer(
            home_gu, rng_comm, mean_degree=community_mean_degree
        ),
    }
    return layers


def network_foi(
    state: np.ndarray,
    layers: dict[str, sparse.csr_matrix],
    beta_by_layer: dict[str, float],
) -> np.ndarray:
    """edge 기반 per-agent 감염력(force of infection)을 계산한다.

    평균장(모두가 구 평균 prevalence 에 노출)과 달리, 각 에이전트의 FoI 는 **자신의
    이웃 중 감염자(STATE_I) 수**에 비례한다. layer ``L`` 에서 에이전트 i 의 기여는
    ``beta_L * (A_L @ infectious)[i]`` — 즉 layer L 의 감염 이웃 수 × 층 전파율.
    여러 layer 는 합산된다. 사망자(STATE_D)는 감염원·수신자 모두에서 제외.

    Args:
        state: 길이 N int 배열. ``agent_kernel`` 의 compartment 코드
            (S=0,E=1,I=2,R=3,V=4,D=5). STATE_I 만 감염원으로 센다.
        layers: ``build_multilayer_network`` 가 반환한 layer→CSR dict. 각 행렬은
            N×N 대칭.
        beta_by_layer: layer 이름→층별 전파율(>=0). ``layers`` 에 있으나 키가 없는
            layer 는 0(미포함)으로 간주.

    Returns:
        길이 N 의 float64 FoI 벡터(>=0). 사망 에이전트 위치는 0. 모든 layer 의
        감염-이웃 가중합.

    Raises:
        ValueError: ``state`` 가 1D 가 아니거나 layer 행렬과 길이가 불일치하거나,
            ``beta_by_layer`` 에 음수/비유한 값이 있는 경우.

    Performance: O(sum_L nnz(A_L)) — 희소 행렬-벡터곱. 평균장 O(N)보다 비싸지만
        edge 수에 선형.
    Side effects: 없음(순수 함수).
    Caller responsibility: ``beta_by_layer`` 값은 일당(per-day) hazard 단위로
        해석된다. kernel 의 ``beta`` 와 동일 스케일이 되도록 캘리브레이션은 호출자
        몫. FoI 를 ``1-exp(-FoI)`` hazard 로 변환하는 것도 호출자 책임.
    """
    state = np.asarray(state)
    if state.ndim != 1:
        raise ValueError(f"state must be 1D; got shape {state.shape}")
    n = state.shape[0]
    infectious = (state == STATE_I).astype(np.float64)
    alive = (state != STATE_D).astype(np.float64)
    foi = np.zeros(n, dtype=np.float64)
    for name, mat in layers.items():
        if mat.shape != (n, n):
            raise ValueError(
                f"layer {name!r} matrix shape {mat.shape} != ({n}, {n})"
            )
        beta = float(beta_by_layer.get(name, 0.0))
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError(
                f"beta_by_layer[{name!r}] must be finite and >= 0; got {beta}"
            )
        if beta == 0.0:
            continue
        # 감염 이웃 수 = A @ infectious. 본인이 죽었으면 수신 0.
        foi += beta * (mat.dot(infectious))
    foi *= alive
    return foi


def sample_infector(
    newly_infected: np.ndarray,
    state: np.ndarray,
    layers: dict[str, sparse.csr_matrix],
    beta_by_layer: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Attribute each newly-infected agent to a specific infector + layer.

    ``network_foi`` collapses neighbour identity in its matrix-vector product, so
    it cannot say WHO infected whom. This companion samples, for each agent that
    just transitioned S→E, one infector from its infectious neighbours across
    layers, weighted by the layer transmission rate ``beta_L`` per infectious edge
    — the same weighting that produced the FoI. This yields the who-infected-whom
    transmission tree (and, aggregated, the offspring / secondary-infection
    distribution and district-to-district transmission).

    Args:
        newly_infected: 1D int array of agent indices that became exposed this step.
        state: length-N compartment codes (``STATE_I`` = infectious source).
        layers: ``build_multilayer_network`` layer→CSR dict (N×N symmetric).
        beta_by_layer: layer→per-day transmission rate (>=0); layers with rate 0
            (or absent) are not eligible infector sources.
        rng: NumPy ``Generator`` (the caller's per-day stream).

    Returns:
        ``(infectors, layer_ids)`` — two length-``len(newly_infected)`` int64
        arrays. ``infectors[k]`` is the sampled infector agent index for
        ``newly_infected[k]``, or ``-1`` when the agent had NO infectious neighbour
        (its infection is attributed to importation). ``layer_ids[k]`` indexes
        ``sorted(layers)`` for the transmitting layer, or ``-1``.

    Performance: O(sum over newly-infected of that agent's degree) — CSR row slices.
    Side effects: consumes ``rng``. Pure otherwise.
    """
    newly_infected = np.asarray(newly_infected, dtype=np.int64)
    m = int(newly_infected.shape[0])
    infectors = np.full(m, -1, dtype=np.int64)
    layer_ids = np.full(m, -1, dtype=np.int64)
    if m == 0:
        return infectors, layer_ids
    infectious = np.asarray(state) == STATE_I
    layer_names = sorted(layers)
    eligible = [(lj, name, float(beta_by_layer.get(name, 0.0)))
                for lj, name in enumerate(layer_names)]
    eligible = [(lj, layers[name], b) for lj, name, b in eligible if b > 0.0]
    for k in range(m):
        i = int(newly_infected[k])
        nbrs, weights, lyrs = [], [], []
        for lj, mat, beta in eligible:
            row = mat.indices[mat.indptr[i]:mat.indptr[i + 1]]  # neighbours of i
            inf_nb = row[infectious[row]]
            if inf_nb.size:
                nbrs.append(inf_nb)
                weights.append(np.full(inf_nb.size, beta, dtype=np.float64))
                lyrs.append(np.full(inf_nb.size, lj, dtype=np.int64))
        if not nbrs:
            continue  # no infectious neighbour → importation (-1)
        nb = np.concatenate(nbrs)
        w = np.concatenate(weights)
        ly = np.concatenate(lyrs)
        choice = int(rng.choice(nb.size, p=w / w.sum()))
        infectors[k] = int(nb[choice])
        layer_ids[k] = int(ly[choice])
    return infectors, layer_ids


def degree_summary(layers: dict[str, sparse.csr_matrix]) -> dict[str, float]:
    """layer 별 평균 degree(1인당 평균 접촉 이웃 수)를 요약한다.

    무방향 대칭 행렬이므로 평균 degree = nnz / N = 행 합의 평균. household layer 는
    가구 크기-1 근처여야 한다(불변식 검증용).

    Args:
        layers: layer 이름→N×N CSR 인접행렬 dict.

    Returns:
        ``{layer_name: mean_degree}`` (float). 모든 layer 에 대해 산출. 추가로
        ``"_total"`` 키에 전 layer 합산(중복 edge 미보정) 평균 degree.

    Raises:
        ValueError: layer 행렬이 정방(N×N)이 아닌 경우.

    Performance: O(num_layers) — 각 행렬의 nnz 는 CSR 메타데이터로 O(1) 조회.
    Side effects: 없음.
    """
    out: dict[str, float] = {}
    total_nnz = 0
    n_ref: int | None = None
    for name, mat in layers.items():
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"layer {name!r} matrix must be square; got {mat.shape}")
        n = mat.shape[0]
        if n_ref is None:
            n_ref = n
        out[name] = float(mat.nnz) / float(n) if n > 0 else 0.0
        total_nnz += mat.nnz
    if n_ref:
        out["_total"] = float(total_nnz) / float(n_ref)
    else:
        out["_total"] = 0.0
    return out
