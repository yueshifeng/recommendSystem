# coding: utf-8
import sys, math, time, random, functools
import numpy as np
from utils import Metrics
from reader import Reader
from numba import jit

class PSO:
    def __init__(self, parameters):
        #parameters = [NGEN, popsize, low, up, featureAndLabel]
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.base = None
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = len(parameters[2]) # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        self.rdd = parameters[4]

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有粒子的位置
        self.pop_v = np.zeros((self.pop_size, self.var_num))  # 所有粒子的速度
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置
        #self.init()

    def init(self):
        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def calc_fusion_score(self, st_p, anctr_p, cardctr_p, cvr_p, ind_var, max_op=False):
        score = 1.0 
        score *= math.pow(1.0 + ind_var[0] * anctr_p, ind_var[1])
        score *= math.pow(1.0 + ind_var[2] * cardctr_p, ind_var[3])
        if max_op:
            score *= math.pow(1 + ind_var[4]* cvr_p * max(anctr_p, cardctr_p), ind_var[5])
        else:
            score *= math.pow(1 + ind_var[4]* cvr_p *(anctr_p + cardctr_p), ind_var[5])
        return score

    def base_auc(self, params=[7.2131, 4.6267, 8.6074, 4.3671, 533.4611, 9.4533], only_sub_result=False, max_op=False):
        # params = [anchor_ctr_c, anchor_ctr_a, card]
        print("=" * 100)
        s = time.time()
        result = self.fitness(params, True, only_sub_result=only_sub_result, max_op=max_op)
        fitness = 0
        if only_sub_result:
            self.base = result
        else:
            fitness = result
        print("base_fitness: %s with time: %s s, with op max_op ? %s" % (fitness, time.time() - s, max_op))
        print("=" * 100)

    @functools.lru_cache()
    def fitness_hash(self, tuple_ind_var, flush_out=True):
        return self.fitness(np.array(tuple_ind_var), flush_out) 
        
    def reward(self, st_auc, anchor_auc, card_auc, cvr_auc):
        if not self.base:
            return anchor_auc * 1 + card_auc * 1.5 + cvr_auc * 10
 
        postives =  [0.0, 0, 0, 6.0]
        negatives = [2.0, 1, 1, 2.0]

        reward_fitness = 0
        base_st_auc, base_anchor_auc, base_card_auc, base_cvr_auc = self.base
        diff = [st_auc - base_st_auc, anchor_auc - base_anchor_auc, card_auc - base_card_auc, cvr_auc - base_cvr_auc]
        for ix, val in enumerate(diff):
            reward_fitness += postives[ix] * diff[ix] if diff[ix] > 0 else negatives[ix] * diff[ix]
        return reward_fitness

    def fitness(self, ind_var, flush_out=True, only_sub_result=False, max_op=True):
        datas = self.rdd
        scores = []
        avg_scores = []
        st_p, st_l, anctr_p, anctr_l, cardctr_p, cardctr_l, cvr_p, cvr_l = [], [], [], [], [], [], [], []
        for data in datas:
            # fetch raw score
            st_p.append(data[0])
            st_l.append(data[1])
            anctr_p.append(data[2])
            anctr_l.append(data[3])
            cardctr_p.append(data[4])
            cardctr_l.append(data[5])
            cvr_p.append(data[6])
            cvr_l.append(data[7])

            # fusion
            score = self.calc_fusion_score(st_p[-1], anctr_p[-1], cardctr_p[-1], cvr_p[-1], ind_var)
            avg_score = self.calc_fusion_score(1.0, anctr_p[-1], cardctr_p[-1], cvr_p[-1], ind_var, max_op=max_op)
            scores.append(score)
            avg_scores.append(avg_score)
        print("avg fusion score is %s with std: %s" % (round(np.mean(avg_scores) ,5), round(np.std(avg_scores), 5)))

        # eval
        st_auc = Metrics.floatLabelAuc(scores, st_l)
        anchor_auc = Metrics.binaryIntLabelAuc(scores, anctr_l) 
        card_auc = Metrics.binaryIntLabelAuc(scores, cardctr_l)
        cvr_auc = Metrics.binaryIntLabelAuc(scores, cvr_l)
        out = 0
        if not only_sub_result:
            out = self.reward(st_auc, anchor_auc, card_auc, cvr_auc)
        if flush_out:
            print("st_auc: %s, anchor_auc: %s, card_auc: %s, cvr_auc: %s, fitness: %s" % (round(st_auc, 4), round(anchor_auc, 4), round(card_auc, 4), round(cvr_auc, 4), round(out, 4) ))
            print("params: %s" % ind_var)
        return out * 1.0 if not only_sub_result else [st_auc, anchor_auc, card_auc, cvr_auc]

    def update_operator(self, pop_size, cur_gen):
        """
        更新算子：更新下一时刻的位置和速度
        """
        c1 = 2  # 学习因子，一般为2
        c2 = 2
        w = 0.5 - (0.5-0.2) * cur_gen / (self.NGEN - 1)
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            pop_fitness = self.fitness_hash(tuple(self.pop_x[i]))
            if pop_fitness > self.fitness_hash(tuple(self.p_best[i])):
                self.p_best[i] = self.pop_x[i]
            if pop_fitness > self.fitness_hash(tuple(self.g_best)):
                self.g_best = self.pop_x[i]
            #if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
            #    self.p_best[i] = self.pop_x[i]
            #if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
            #    self.g_best = self.pop_x[i]

    def main(self):
        self.init()
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        maxVal = 0
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size, gen)
            popobj.append(self.fitness_hash(tuple(self.g_best)))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness_hash(tuple(self.g_best)) > self.fitness_hash(tuple(self.ng_best)):
                self.ng_best = self.g_best.copy()
            maxVal = self.fitness_hash(tuple(self.ng_best))
            print('最好的位置：{}'.format(self.ng_best))
            print('最大的函数值：{}'.format(maxVal))
        print("---- End of (successful) Searching ----")
        return (maxVal, self.ng_best)

def psoProcess(featureAndLabel):
    NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    popsize = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    low = [0.00001,1, 0.00001, 1, 0.0000001, 1] 
    up = [10, 10, 10, 10, 10000, 20]
    parameters = [NGEN, popsize, low, up, featureAndLabel]
    pso = PSO(parameters)

    pso.base_auc(only_sub_result=False, max_op=False)
    #pso.base_auc(params=[10, 5.26218864, 4.31611505, 6.82562373,193.3424281,17.56299594], only_sub_result=False, max_op=False)
    pso.base_auc(params=[2.2672, 6.5901, 7.5287, 4.4361, 524.591, 13.8014], only_sub_result=False, max_op=False)
    #return pso.main()

if __name__ == "__main__":
    featureAndLabels = Reader("auc_spark2").parseLines(sample_rate=0.1)
    psoProcess(featureAndLabels)
 
