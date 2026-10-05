import os
from typing import List

from mealpy.utils.agent import Agent

import time

import numpy as np
from mealpy import FloatVar, PSO, Optimizer


# 设置CEC函数参数
BENCHMARK = "CEC2022"
DIM = 20  # CEC2022支持的维度：2、10、20
LB = -100  # 变量下界
UB = 100  # 变量上界
FEs = 1000000
POP = 50   # 种群大小
GEN = int(FEs / POP)  # 迭代次数


class CustomCLPSO(PSO.CL_PSO):
    def __init__(self, epoch, pop_size, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)

    def track_optimize_step(self, pop: List[Agent] = None, epoch: int = None, runtime: float = None) -> None:
        """
        重写父类的 track_optimize_step 方法，将多样性计算改为基于均值中心的距离。
        """
        ## convergence. 保存常规历史数据 (保留原逻辑)
        if self.problem.save_population:
            self.history.list_population.append(Optimizer.duplicate_pop(pop))
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1].target.fitness)
        self.history.list_current_best_fit.append(self.history.list_current_best[-1].target.fitness)

        ## 2. [修改部分] 计算多样性 (改为：平均中心距离)
        # 原代码:
        # pos_matrix = np.array([agent.solution for agent in pop])
        # div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        # self.history.list_diversity.append(np.mean(div, axis=0))

        # 新代码: 基于均值中心 (Mean Center) 和 欧几里得距离 (Euclidean Distance)
        try:
            pos_matrix = np.array([agent.solution for agent in pop])
            # 计算种群中心 (均值)
            center = np.mean(pos_matrix, axis=0)
            # 计算每个粒子到中心的欧几里得距离 (L2范数)
            distances = np.linalg.norm(pos_matrix - center, axis=1)
            # 计算平均距离作为多样性指标
            diversity_metric = np.mean(distances)

            self.history.list_diversity.append(diversity_metric)
        except Exception as e:
            print(f"Error calculating diversity: {e}")
            self.history.list_diversity.append(0)  # 出错时填充0防止对齐错误

        ## 3. 打印日志 (保留原逻辑)
        if epoch % 1000 == 0:
            self.logger.info(f">>>Problem: {self.problem.name}, Epoch: {epoch}, "
                         f"Current best: {self.history.list_current_best[-1].target.fitness:.6f}, "
                         f"Global best: {self.history.list_global_best[-1].target.fitness:.6f}, "
                         f"Runtime: {runtime:.5f} seconds")

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

def main(func_num, benchmark=BENCHMARK):
    global BENCHMARK, DIM, FEs, GEN
    BENCHMARK = str(benchmark).upper()
    if BENCHMARK not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}")
    DIM = 30 if BENCHMARK == "CEC2017" else 20
    FEs = 10000 * DIM if BENCHMARK == "CEC2017" else 1000000
    GEN = FEs // POP
    def objective_function(x):
        return evaluate_scalar(x, func_num, DIM, BENCHMARK)

    problem_dict = {
        "bounds": FloatVar(lb=(LB,) * DIM, ub=(UB,) * DIM),
        "obj_func": objective_function,
        "minmax": "min",
    }

    model = CustomCLPSO(epoch=GEN, pop_size=POP, c_local = 1.49445, w_min=0.4, w_max=0.9, max_flag = 7)
    time_list = []
    for seed in range(30):

        start_time = time.time()
        g_best = model.solve(problem_dict, seed=seed)
        end_time = time.time()
        time_val = end_time - start_time
        time_list.append(time_val)
        write_div(model.history.list_diversity, 'CLPSO', func_num)
        write_log(model.history.list_global_best_fit, 'CLPSO', func_num)
        print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    write_time(time_list, 'CLPSO')
