from __future__ import annotations

import os

import time
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple
import numpy as np

Array = np.ndarray

BENCHMARK = "CEC2022"
DIM = 20  # CEC2022支持的维度：2、10、20
LB = -100  # 变量下界
UB = 100  # 变量上界
FEs = 1000000
POP = 50   # 种群大小
GEN = int(FEs / POP)  # 迭代次数

def cec2022_objective(x, func_num):
    return evaluate(x, func_num, DIM, BENCHMARK)[0]

def write_log(gbest_list, name, func_num):
    log_dir = result_dir(BENCHMARK, DIM, "convergence", func_num)
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    log_file = os.path.join(log_dir, name)

    # 构造一行：代与代之间用制表符分隔，保留6位小数
    line = ','.join([f"{val:.3f}" for val in gbest_list]) + '\n'

    # 以追加方式写入文件，每行是一次运行
    with open(log_file, "a") as f:
        f.write(line)

def write_div(div_list, name, func_num):
    log_dir = result_dir(BENCHMARK, DIM, "diversity", func_num)
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    log_file = os.path.join(log_dir, name)

    # 构造一行：代与代之间用制表符分隔，保留6位小数
    line = ','.join([f"{val:.3f}" for val in div_list]) + '\n'

    # 以追加方式写入文件，每行是一次运行
    with open(log_file, "a") as f:
        f.write(line)

def write_time(time_list, name):
    log_dir = result_dir(BENCHMARK, DIM, "time")
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    log_file = os.path.join(log_dir, name)

    # 构造一行：代与代之间用制表符分隔，保留6位小数
    line = ','.join([f"{val:.3f}" for val in time_list]) + '\n'

    # 以追加方式写入文件，每行是一个函数
    with open(log_file, "a") as f:
        f.write(line)

@dataclass
class TAPSOResult:
    gbest: Array
    gbestval: float
    fitcount: int
    suc: int
    suc_fes: int


def tapso_func(
    seed: int,
    jingdu: float,
    fhd: Callable[[Array, Sequence[float]], float] | Callable[[Array], float],
    max_gen: int,
    max_fes: int,
    particle_number: int,
    dimension: int,
    vrmin: float | Sequence[float],
    vrmax: float | Sequence[float],
    *fhd_args: float,
) -> TAPSOResult:
    rng = np.random.default_rng(seed)  # rand('state', seed)

    me = max_gen  # me=Max_Gen;
    ps = particle_number  # ps=Particle_Number;
    d = dimension  # D=Dimension;

    if isinstance(vrmin, (int, float)):
        vrmin_arr = np.full(d, vrmin, dtype=float)
    else:
        vrmin_arr = np.array(vrmin, dtype=float)
    if isinstance(vrmax, (int, float)):
        vrmax_arr = np.full(d, vrmax, dtype=float)
    else:
        vrmax_arr = np.array(vrmax, dtype=float)

    vrmin_mat = np.repeat(vrmin_arr[np.newaxis, :], ps, axis=0)  # VRmin=repmat(VRmin,ps,convergence);
    vrmax_mat = np.repeat(vrmax_arr[np.newaxis, :], ps, axis=0)  # VRmax=repmat(VRmax,ps,convergence);
    mv = 0.5 * (vrmax_mat - vrmin_mat)  # mv=0.5.*(VRmax-VRmin);

    pos = vrmin_mat + (vrmax_mat - vrmin_mat) * rng.random((ps, d))  # 初始化粒子位置
    vel = -mv + 2.0 * mv * rng.random((ps, d))  # vel = -mv + 2 .* mv .* rand(ps, D);

    neighbor = np.zeros((ps, 2), dtype=int)  # ring topology neighbor(i,:)=[i-convergence,i+convergence];
    neighbor[0, :] = np.array([ps - 1, 1])
    for i in range(1, ps - 1):
        neighbor[i, :] = np.array([i - 1, i + 1])
    neighbor[ps - 1, :] = np.array([ps - 2, 0])

    e = np.zeros((ps, 1))  # 记录每个个体的适应度 e(i,convergence)=feval(...);
    for i in range(ps):
        e[i, 0] = _call_fhd(fhd, pos[i, :], fhd_args)
    fitcount = ps

    pbestval = e.copy()  # pbestval(i,convergence)=e(i,convergence);
    pbest = pos.copy()
    old_e = e.copy()
    old_pos = pos.copy()
    old_pbestval = pbestval.copy()
    old_pbest = pbest.copy()

    iwt = np.full(d, 0.7298)  # iwt=0.7298*ones(convergence,D);
    cc = np.array([1.0, 1.0, 1.0])  # cc=[convergence.0, convergence.0, convergence.0];

    gbestid = np.argmin(pbestval)
    gbest = pbest[gbestid, :].copy()
    gbestval = float(pbestval[gbestid, 0])

    for k in range(ps):  # 与 MATLAB 预热相同
        nb_ids = neighbor[k, :]
        tmpid = np.argmin(pbestval[nb_ids])
        aa = (
            cc[0] * rng.random(d) * (pbest[k, :] - pos[k, :])
            + cc[1] * rng.random(d) * (pbest[nb_ids[tmpid], :] - pos[k, :])
        )
        vel[k, :] = iwt * vel[k, :] + aa
        vel[k, :] = np.clip(vel[k, :], -mv[k, :], mv[k, :])

        pos[k, :] = pos[k, :] + vel[k, :]
        pos[k, :] = np.clip(pos[k, :], vrmin_mat[k, :], vrmax_mat[k, :])

        e[k, 0] = _call_fhd(fhd, pos[k, :], fhd_args)
        fitcount += 1

        if e[k, 0] < pbestval[k, 0]:
            pbestval[k, 0] = e[k, 0]
            pbest[k, :] = pos[k, :]
            if e[k, 0] < gbestval:
                gbestval = float(e[k, 0])
                gbest = pbest[k, :].copy()

    ratio = 0.5  # 加入队列的优势个体的比例
    upper_elite = ps
    lower_elite = round(ps / 2)
    elites = round(ps / 4)
    sizeof_af = round(ps / 4)  # sizeof_Af
    sizeof_adf = round(ps / 4)  # sizeof_Adf
    sizeof_adfdd = round(ps / 4)  # sizeof_Adfdd
    archive_maxsize = ps  # archive_maxsize = ps;

    (
        q_f,
        q_f_pos,
        prob_f,
        q_df,
        q_df_pos,
        prob_df,
        q_df_dd,
        q_df_dd_pos,
        prob_df_dd,
    ) = _initialize_queues(
        pbest,
        pbestval,
        old_pbest,
        old_pbestval,
        sizeof_af,
        sizeof_adf,
        sizeof_adfdd,
    )

    #diversity = np.zeros(max_fes, dtype=float)  # 用于记录“计算多样性 --- 测试用 ---”部分
    div_list = []
    old2 = 0
    new2 = fitcount
    diversity = _compute_diversity(q_df_dd_pos)
    div_list.append(diversity)

    old_pbestval = pbestval.copy()
    old_pbest = pbest.copy()

    archieve_val = np.array([], dtype=float)  # archieve_val=[]
    archieve_pos = np.empty((0, d))  # archieve_pos=[]

    suc = 0
    suc_fes = 0
    iter_count = 1
    ceased = 0
    gbest_list = [0.0] * int(GEN/2)
    gbest_list[0] = gbestval
    while fitcount < max_fes:  # 主要循环，与 MATLAB while fitcount<Max_FES 对应
        iter_count += 1
        ix = np.argsort(pbestval[:, 0])

        for k in range(ps):
            j = ix[k]
            obj_x, prob_f, prob_df, prob_df_dd = select(
                k,
                ps,
                elites,
                q_f,
                q_f_pos,
                prob_f,
                q_df,
                q_df_pos,
                prob_df,
                q_df_dd,
                q_df_dd_pos,
                prob_df_dd,
                vrmax_arr,
                vrmin_arr,
                rng,
            )

            obj_val = _call_fhd(fhd, obj_x[0, :], fhd_args)
            fitcount += 1

            if obj_val <= gbestval:  # 对应（3）
                archieve_val, archieve_pos = add2archieve(
                    archieve_val, archieve_pos, archive_maxsize, obj_x, obj_val
                )
                iwt_local = 0.45 + 0.005 * np.tan(np.pi * (rng.random(d) - 0.5))
                vel[j, :] = iwt_local * vel[j, :] + cc[2] * rng.random(d) * (
                    obj_x[0, :] - pos[j, :]
                )
                gbest = obj_x[0, :].copy()
                gbestval = float(obj_val)
            elif obj_val <= pbestval[j, 0]:  # 对应（2）
                archieve_val, archieve_pos = add2archieve(
                    archieve_val, archieve_pos, archive_maxsize, obj_x, obj_val
                )
                iwt_local = 0.6 + 0.005 * np.tan(np.pi * (rng.random(d) - 0.5))
                vel[j, :] = (
                    iwt_local * vel[j, :]
                    + cc[0] * rng.random(d) * (obj_x[0, :] - pos[j, :])
                    + cc[1] * rng.random(d) * (gbest - pos[j, :])
                )
            else:  # 对应（convergence）
                iwt_local = 0.75 + 0.005 * np.tan(np.pi * (rng.random(d) - 0.5))
                vel[j, :] = (
                    iwt_local * vel[j, :]
                    + cc[0] * rng.random(d) * (pbest[j, :] - pos[j, :])
                    + cc[1] * rng.random(d) * (gbest - pos[j, :])
                )

            vel[j, :] = np.clip(vel[j, :], -mv[0, :], mv[0, :])
            pos[j, :] = pos[j, :] + vel[j, :]
            pos[j, :] = np.clip(pos[j, :], vrmin_arr, vrmax_arr)

            if rng.random() >= 0.5:
                lower = vrmin_arr + 0.1 * (vrmax_arr - vrmin_arr) * rng.random(d)
                upper = vrmax_arr - 0.1 * (vrmax_arr - vrmin_arr) * rng.random(d)
                mask_valid = (pos[j, :] >= vrmin_arr) & (pos[j, :] <= vrmax_arr)
                mask_lower = pos[j, :] < vrmin_arr
                mask_upper = pos[j, :] > vrmax_arr
                pos_j = np.zeros_like(pos[j, :])
                pos_j[mask_valid] = pos[j, mask_valid]
                pos_j[mask_lower] = lower[mask_lower]
                pos_j[mask_upper] = upper[mask_upper]
                pos[j, :] = pos_j

            e[j, 0] = _call_fhd(fhd, pos[j, :], fhd_args)
            fitcount += 1

            if e[j, 0] < pbestval[j, 0]:
                pbestval[j, 0] = e[j, 0]
                pbest[j, :] = pos[j, :]
                if pbestval[j, 0] < gbestval:
                    gbestval = float(pbestval[j, 0])
                    gbest = pbest[j, :].copy()

        old2 = new2  # 计算多样性 --- 测试用 ---
        new2 = min(fitcount, max_fes)
        diversity = _compute_diversity(q_df_dd_pos)
        div_list.append(diversity)

        pbest, pbestval, vel = reuse_exemplars(
            pbest, pbestval, vel, archieve_pos, archieve_val, rng
        )

        best_idx = np.argmin(pbestval[:, 0])
        if pbestval[best_idx, 0] < gbestval:
            gbest = pbest[best_idx, :].copy()
            gbestval = float(pbestval[best_idx, 0])
            ceased = 0
        else:
            ceased += 1

        pbestval_df = old_pbestval - pbestval
        df = old_pbestval - pbestval
        dd = np.linalg.norm(pbest - old_pbest, axis=1, keepdims=True)
        dd_safe = np.where(dd == 0, np.finfo(float).tiny, dd)
        df_dd = df / dd_safe

        (
            q_f,
            q_f_pos,
            prob_f,
            q_df,
            q_df_pos,
            prob_df,
            q_df_dd,
            q_df_dd_pos,
            prob_df_dd,
        ) = update_q(
            q_f,
            q_f_pos,
            prob_f,
            q_df,
            q_df_pos,
            prob_df,
            q_df_dd,
            q_df_dd_pos,
            prob_df_dd,
            pbestval,
            pbest,
            pbestval_df,
            df_dd,
            pos,
        )

        old_pbestval = pbestval.copy()
        old_pbest = pbest.copy()
        gbest_list[iter_count-1] = gbestval
        if iter_count % 1000 == 0:
            print(f"Gen: {iter_count}, Best Value: {gbestval} ")
        if fitcount >= max_fes:
            break
    write_div(div_list, "TAPSO", fhd_args[0])
    write_log(gbest_list,"TAPSO", *fhd_args)
    result = TAPSOResult(
        gbest=gbest.copy(),
        gbestval=float(gbestval),
        fitcount=int(fitcount),
        suc=suc,
        suc_fes=suc_fes,
    )
    return result


def _call_fhd(
    fhd: Callable[[Array], float] | Callable[[Array, Sequence[float]], float],
    x: Array,
    args: Sequence[float],
) -> float:
    if args:
        return float(fhd(x, *args))
    return float(fhd(x))


def _initialize_queues(
    pbest: Array,
    pbestval: Array,
    old_pbest: Array,
    old_pbestval: Array,
    sizeof_af: int,
    sizeof_adf: int,
    sizeof_adfdd: int,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """对应 MATLAB 初始化三类优先队列的三段代码。"""
    ps, d = pbest.shape
    tmppbest = np.vstack((old_pbest, pbest))
    tmppbestval = np.vstack((old_pbestval, pbestval))
    order = np.argsort(tmppbestval[:, 0])
    tmppbest_sorted = tmppbest[order]
    tmppbestval_sorted = tmppbestval[order]

    q_f_pos = []
    q_f_val = []
    seen = set()
    for row, val in zip(tmppbest_sorted, tmppbestval_sorted[:, 0]):
        key = tuple(row.round(decimals=12))
        if key in seen:
            continue
        seen.add(key)
        q_f_pos.append(row.copy())
        q_f_val.append(val)
        if len(q_f_pos) >= sizeof_af:
            break

    q_f_pos = np.array(q_f_pos)
    q_f = np.array(q_f_val).reshape(-1, 1)
    prob_f = _normalize_prob(np.arange(len(q_f), 0, -1, dtype=float))

    pbestval_dif = old_pbestval - pbestval
    order_df = np.argsort(-pbestval_dif[:, 0])
    tmppbest_df = pbest[order_df]
    tmppbestval_df = pbestval_dif[order_df]

    q_df_pos = tmppbest_df[:sizeof_adf]
    q_df = tmppbestval_df[:sizeof_adf]
    prob_df = _normalize_prob(np.arange(len(q_df), 0, -1, dtype=float))

    df = old_pbestval - pbestval
    dd = np.linalg.norm(pbest - old_pbest, axis=1, keepdims=True)
    dd_safe = np.where(dd == 0, np.finfo(float).tiny, dd)
    tmp_df_dd = (df / dd_safe).flatten()
    order_dfdd = np.argsort(-tmp_df_dd)
    q_df_dd_pos = pbest[order_dfdd[:sizeof_adfdd]]
    q_df_dd = tmp_df_dd[order_dfdd[:sizeof_adfdd]].reshape(-1, 1)
    prob_df_dd = _normalize_prob(np.arange(len(q_df_dd), 0, -1, dtype=float))

    return (
        q_f,
        q_f_pos,
        prob_f,
        q_df,
        q_df_pos,
        prob_df,
        q_df_dd,
        q_df_dd_pos,
        prob_df_dd,
    )


def _compute_diversity(q_df_dd_pos: Array) -> float:
    # if q_df_dd_pos.size == 0:
    #     return 0.0
    # center = np.mean(q_df_dd_pos, axis=0)
    # div = 0.0
    # for row in q_df_dd_pos:
    #     div += np.sum((row - center) ** 2)
    # return div / (q_df_dd_pos.shape[0] * (vrmax[0] - vrmin[0]))
    # convergence. 安全检查：如果种群为空，返回 0
    if q_df_dd_pos.size == 0:
        return 0.0

    # 2. 计算种群中心 (Centroid)
    # axis=0 对每一列求均值，得到中心点坐标
    center = np.mean(q_df_dd_pos, axis=0)

    # 3. 计算每个粒子到中心的欧几里得距离 (L2 Norm)
    # axis=convergence 对每一行（每个粒子向量）计算长度
    # 这一步替代了原来的 (row - center)**2 求和循环
    distances = np.linalg.norm(q_df_dd_pos - center, axis=1)

    # 4. 返回平均距离
    # 这一步替代了原来的 div / (N * range)
    return np.mean(distances)


def add2archieve(
    archieve_val: Array,
    archieve_pos: Array,
    max_size: int,
    obj_x: Array,
    obj_val: float,
) -> Tuple[Array, Array]:
    """对应 add2archieve.m 子函数。"""
    obj_vec = obj_x[0, :].copy()
    if archieve_val.size < max_size:
        archieve_val = np.append(archieve_val, obj_val)
        if archieve_pos.size == 0:
            archieve_pos = obj_vec.reshape(1, -1)
        else:
            archieve_pos = np.vstack((archieve_pos, obj_vec))
    else:
        idx = np.argmax(archieve_val)
        archieve_val[idx] = obj_val
        archieve_pos[idx, :] = obj_vec
    return archieve_val, archieve_pos


def reuse_exemplars(
    pbest: Array,
    pbestval: Array,
    vel: Array,
    archieve_pos: Array,
    archieve_val: Array,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Array]:
    """对应 Reuse_Exemplars 函数，实现档案重用。"""
    archiev_size = len(archieve_val)
    if archiev_size == 0:
        return pbest, pbestval, vel

    ps, d = pbest.shape
    val_p = np.sort(pbestval[:, 0])[::-1]
    index_p = np.argsort(pbestval[:, 0])[::-1]

    rp = rng.permutation(archiev_size)
    tmpval = []
    tmppos = []
    for i in range(0, archiev_size - 1, 2):
        first = rp[i]
        second = rp[i + 1]
        if archieve_val[first] < archieve_val[second]:
            tmpval.append(archieve_val[first])
            tmppos.append(archieve_pos[first, :].copy())
        else:
            tmpval.append(archieve_val[second])
            tmppos.append(archieve_pos[second, :].copy())

    tmpval = np.array(tmpval)
    tmppos = np.array(tmppos)

    i_p = 0
    index = 0
    while i_p < ps and index < len(tmpval):
        if val_p[i_p] > tmpval[index] and rng.random() < 0.5:
            pbestval[index_p[i_p], 0] = tmpval[index]
            pbest[index_p[i_p], :] = tmppos[index, :].copy()
            vel[index_p[i_p], :] = np.zeros(d)
            i_p += 1
        index += 1

    return pbest, pbestval, vel


def select(
    i: int,
    ps: int,
    elites: int,
    queue_f: Array,
    pos_f: Array,
    prob_f: Array,
    queue_df: Array,
    pos_df: Array,
    prob_df: Array,
    queue_df_dd: Array,
    pos_df_dd: Array,
    prob_df_dd: Array,
    xmax: Array,
    xmin: Array,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Array, Array]:
    """对应 select 函数，为个体选择学习榜样。"""
    obj_x, prob_f_out, prob_df_out = generate(
        queue_f,
        prob_f,
        pos_f,
        queue_df_dd,
        prob_df_dd,
        pos_df_dd,
        xmax,
        xmin,
        rng,
    )
    return obj_x, prob_f_out, prob_df, prob_df_out


def generate(
    queue_1: Array,
    prob_1: Array,
    pos_1: Array,
    queue_2: Array,
    prob_2: Array,
    pos_2: Array,
    xmax: Array,
    xmin: Array,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Array]:
    """对应 Generate 函数，逐维/整体算术杂交+变异。"""
    n = queue_1.shape[0]
    d = pos_1.shape[1]
    pc = 0.5
    pm = 0.02

    cum_prob1 = np.cumsum(prob_1)
    cum_prob2 = np.cumsum(prob_2)

    exemplar = np.zeros((1, d))
    for j in range(d):
        idx1 = min(np.searchsorted(cum_prob1, rng.random(), side="right"), n - 1)
        idx2 = min(np.searchsorted(cum_prob2, rng.random(), side="right"), n - 1)
        x1 = pos_1[idx1, :]
        x2 = pos_2[idx2, :]
        rd = rng.random() < pc
        exemplar[0, j] = rd * x1[j] + (1 - rd) * x2[j]

    for j in range(d):
        if rng.random() < pm:
            exemplar[0, j] = xmin[j] + (xmax[j] - xmin[j]) * rng.random()

    return exemplar, prob_1, prob_2


def update_q(
    q_f: Array,
    q_f_pos: Array,
    prob_f: Array,
    q_df: Array,
    q_df_pos: Array,
    prob_df: Array,
    q_df_dd: Array,
    q_df_dd_pos: Array,
    prob_df_dd: Array,
    pbestval: Array,
    pbest: Array,
    pbestval_df: Array,
    df_dd: Array,
    pos: Array,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """对应 Update_Q 函数，刷新三类队列与概率。"""
    order = np.argsort(pbestval[:, 0])
    tmppbestval = pbestval[order]
    tmppbest = pbest[order]

    ps = q_f.shape[0]
    tmp_q_f = []
    tmp_q_f_pos = []
    i = j = 0
    while len(tmp_q_f) < ps and (i < len(tmppbestval) or j < len(q_f)):
        take_new = j >= len(q_f) or (i < len(tmppbestval) and tmppbestval[i, 0] < q_f[j, 0])
        if take_new:
            candidate = tmppbest[i, :]
            if not _contains(tmp_q_f_pos, candidate):
                tmp_q_f.append(tmppbestval[i, 0])
                tmp_q_f_pos.append(candidate.copy())
            i += 1
        else:
            candidate = q_f_pos[j, :]
            if not _contains(tmp_q_f_pos, candidate):
                tmp_q_f.append(q_f[j, 0])
                tmp_q_f_pos.append(candidate.copy())
            j += 1

    q_f = np.array(tmp_q_f).reshape(-1, 1)
    q_f_pos = np.array(tmp_q_f_pos)
    prob_f = _normalize_prob(np.arange(len(q_f), 0, -1, dtype=float))

    order_df = np.argsort(-pbestval_df[:, 0])
    q_df = pbestval_df[order_df][: q_df.shape[0]]
    q_df_pos = pbest[order_df][: q_df_pos.shape[0]]
    prob_df = _normalize_prob(np.arange(len(q_df), 0, -1, dtype=float))

    df_dd_flat = df_dd.flatten()
    order_dfdd = np.argsort(-df_dd_flat)
    q_df_dd = df_dd_flat[order_dfdd][: q_df_dd.shape[0]].reshape(-1, 1)
    q_df_dd_pos = pos[order_dfdd][: q_df_dd_pos.shape[0]]
    prob_df_dd = _normalize_prob(np.arange(len(q_df_dd), 0, -1, dtype=float))

    return (
        q_f,
        q_f_pos,
        prob_f,
        q_df,
        q_df_pos,
        prob_df,
        q_df_dd,
        q_df_dd_pos,
        prob_df_dd,
    )


def _contains(container: Sequence[Array], candidate: Array) -> bool:
    if len(container) == 0:
        return False
    return any(np.allclose(row, candidate) for row in container)


def _normalize_prob(prob: Array) -> Array:
    total = prob.sum()
    if total == 0:
        return prob
    return prob / total

def main(func_num, benchmark=BENCHMARK):
    global BENCHMARK, DIM, FEs, GEN
    BENCHMARK = str(benchmark).upper()
    if BENCHMARK not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}")
    DIM = 30 if BENCHMARK == "CEC2017" else 20
    FEs = 10000 * DIM if BENCHMARK == "CEC2017" else 1000000
    GEN = FEs // POP
    time_list =[]
    for seed in range(1, 30):
        start_time = time.time()
        result = tapso_func(seed,1e-8,cec2022_objective,GEN,FEs,POP,DIM,LB,UB,func_num)
        end_time = time.time()
        time_val = end_time - start_time
        time_list.append(time_val)
        print(f"Func {func_num}, Seed {seed}: gbestval={result.gbestval}, fitcount={result.fitcount}")
    write_time(time_list, 'TAPSO')
