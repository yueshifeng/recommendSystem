#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from gaussian_model_utils import bound_x, seek_model, train_seed_cnt, pop_size, NGEN, w_init, w_final, \
    c1, c2, GAUSSIAN_NGEN, TRAIN_DATA_IGNORE_SUPPRESS, TRAIN_DATA_SIZE, FINISH_LIKE_COMMENT_FIX, GAUC_MIN_DATA_SIZE, \
    GAUC_MAX_DATA_SIZE, START_DAY, END_DAY, sql, header, reward
from math import sin
from math import pi
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
import random
import multiprocessing
import argparse
import base64
import copy
from video_util.common import *
from video_util.feature import *

importPySpark()
sparkContext, sparkSession = initPySpark(enableHiveSupport=True)
from pyspark.sql import SQLContext, HiveContext

PROTOBUF_FILE = "hdfs://R2/projects/ci_rcmd/hdfs/dev/recommend/yarong_video/de_user_profile/proto/video_user_profile_pb2.py"
REDIS_PACKAGE = "hdfs://R2/projects/ci_rcmd/hdfs/udf/resources/redis.zip"
SPARK_PACKAGE = "hdfs://R2/projects/ci_rcmd/hdfs/udf/resources/spark-tfrecord_2.11-0.3.4.jar"
partition_num = 600

pop_x = []
pop_v = []
p_best = []
p_best_result = [-1 for i in range(pop_size)]
g_best_result = -1
g_best = {}
g_best_result_list = []
g_best_position_list = []
base_gauc_detail = ''

for model in bound_x:
    g_best[model] = [0., 0., 0.]

class Hive(object):

    def __init__(self):
        self.spark_session = sparkSession
        self.spark_context = sparkContext
        self.sql_context = SQLContext(self.spark_context)
        self.hive_context = HiveContext(self.sql_context)

    def query(self, sql):
        '''
        when exception happened, empty list will be returned
        '''
        query_result = []
        try:
            query_result = self.hive_context.sql(sql).rdd.collect()
        except Exception as e:
            print('exec sql on spark failed: sql={}, exception={}'.format(sql, str(e)))
        return query_result

    def stop(self):
        self.spark_session.stop()


def normalizeReqId(reqid):
    return reqid.replace(":", "%3A")


logging.basicConfig(level=logging.ERROR)


def get_x_sample_data(seed_cnt, X):
    X_VAL = []
    X_PARAM = []
    for i in range(seed_cnt):
        for train_sample in X[-10:]:
            params_x = {}
            x_per_seed = []
            sum_a = 0
            cnt = 0
            for model_index, model in enumerate(bound_x):
                lower = bound_x[model]['lower']
                upper = bound_x[model]['upper']
                if cnt == seed_cnt % 9:
                    a_x = min(max(round(train_sample[model_index * 3] + random.uniform(-1,1), 4), lower[0]), upper[0])
                    b_x = min(max(round(train_sample[model_index * 3 + 1] + random.uniform(-1,1), 4), lower[1]), upper[1])
                    c_x = min(max(round(train_sample[model_index * 3 + 2] + random.uniform(-1,1), 4), lower[1]), upper[2])
                else:
                    a_x = train_sample[model_index * 3]
                    b_x = train_sample[model_index * 3 + 1]
                    c_x = train_sample[model_index * 3 + 2]
                # a_x = round(random.uniform(lower[0], upper[0]), 4)
                # b_x = round(random.uniform(lower[1], upper[1]), 4)
                # c_x = round(random.uniform(lower[2], upper[2]), 4)
                params_x[model] = [a_x, b_x, c_x]
                x_per_seed.append(a_x)
                x_per_seed.append(b_x)
                x_per_seed.append(c_x)
                sum_a += a_x
                cnt += 1
            if sum_a > 30:
                for model_index, model in enumerate(bound_x):
                    params_x[model][0] = params_x[model][0] * 30.0 / sum_a
                    x_per_seed[model_index * 3] = round(x_per_seed[model_index * 3] * 30.0 / sum_a, 4)
            X_VAL.append(x_per_seed)
            X_PARAM.append(params_x)
    return X_VAL, X_PARAM


# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)


# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    # mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs


# optimize the acquisition function
def opt_acquisition(X, model):
    # random search, generate random samples
    Xsamples = get_x_sample_data(train_seed_cnt, X)
    # Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples[0], model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[0][ix], Xsamples[1][ix]


def init_param(val_list, df, is_coin_user):
    global pop_x
    global pop_v
    global p_best
    global p_best_result
    global g_best_result
    global g_best
    global g_best_result_list
    global g_best_position_list
    global base_gauc_detail
    for i in range(pop_size):
        params_x = {}
        params_v = {}
        p_best_i = {}
        x_per_seed = []
        for model in bound_x:
            lower = bound_x[model]['lower']
            upper = bound_x[model]['upper']
            model_type = bound_x[model]['model_type']
            print("init model_type:{},{}".format(model_type, model))
            # if (model_type == 'like' or model_type == 'comment') and FINISH_LIKE_COMMENT_FIX:
            #     lower = bound_x[model]['param']
            #     upper = bound_x[model]['param']
            if i == 0:
                if is_coin_user:
                    a_x = bound_x[model]['coin_param'][0]
                    b_x = bound_x[model]['coin_param'][1]
                    c_x = bound_x[model]['coin_param'][2]
                else:
                    a_x = bound_x[model]['param'][0]
                    b_x = bound_x[model]['param'][1]
                    c_x = bound_x[model]['param'][2]
            # elif i == 1:
            #     a_x = bound_x[model]['exp_param'][0]
            #     b_x = bound_x[model]['exp_param'][1]
            #     c_x = bound_x[model]['exp_param'][2]
            # elif i == 2:
            #     a_x = bound_x[model]['base_param'][0]
            #     b_x = bound_x[model]['base_param'][1]
            #     c_x = bound_x[model]['base_param'][2]
            else:
                a_x = round(random.uniform(lower[0], upper[0]), 4)
                b_x = round(random.uniform(lower[1], upper[1]), 4)
                c_x = round(random.uniform(lower[2], upper[2]), 4)
            a_v = round(random.uniform(0, 1), 4)
            b_v = round(random.uniform(0, 1), 4)
            c_v = round(random.uniform(0, 1), 4)
            print("index:{}, is_coin_user: {},init model_value:{},{}".format(i, is_coin_user, a_x, b_x, c_x))
            params_x[model] = [a_x, b_x, c_x]
            params_v[model] = [a_v, b_v, c_v]
            p_best_i[model] = [0., 0., 0.]
            x_per_seed.append(a_x)
            x_per_seed.append(b_x)
            x_per_seed.append(c_x)
        pop_x.append(params_x)
        pop_v.append(params_v)
        p_best.append(p_best_i)

    for i in range(pop_size):
        p_best[i] = pop_x[i]
        fit = reward(val_list, df, p_best[i], TRAIN_DATA_IGNORE_SUPPRESS, True if i == 0 else False, is_coin_user)
        p_best_result[i] = fit[0]
        global g_best_result
        global g_best
        g_best_position_list.append(copy.deepcopy(p_best[i]))
        g_best_result_list.append(copy.deepcopy(fit[0]))
        if fit[0] > g_best_result:
            g_best = copy.deepcopy(p_best[i])
            g_best_result = copy.deepcopy(fit[0])


def update_operator(val_list, df, cur_gen, is_coin_user):
    global pop_x
    global pop_v
    global p_best
    global p_best_result
    global g_best_result
    global g_best
    global g_best_result_list
    global g_best_position_list
    global base_gauc_detail
    begin_time = time.time()
    w = 0.9 - (0.9 - 0.4) * cur_gen / (NGEN - 1)
    for i in range(pop_size):
        c = 0 if p_best_result[i] == -1 else 1
        sum_a = 0
        for model in bound_x:
            for loc in range(len(pop_v[i][model])):
                print("$$$$increase rate$$$$:{},{}".format(c, model))
                pop_v[i][model][loc] = round(
                    w * pop_v[i][model][loc] + (1 - w) * (c * (p_best[i][model][loc] - pop_x[i][model][loc])
                                                          + (g_best[model][loc] - pop_x[i][model][loc])), 4)
                pop_x[i][model][loc] = round(max(min(pop_x[i][model][loc] + pop_v[i][model][loc],
                                                     bound_x[model]["upper"][loc]), bound_x[model]["lower"][loc]), 4)
                if loc == 0:
                    sum_a += pop_x[i][model][loc]
        if sum_a > 30:
            for model in bound_x:
                pop_x[i][model][0] = round(pop_x[i][model][0] * 30.0 / sum_a, 4)
            print("$$$$sum_a$$$$:{},{}".format(sum_a, pop_x[i]))
        ind_result = reward(val_list, df, pop_x[i], TRAIN_DATA_IGNORE_SUPPRESS, False, is_coin_user)
        print('pso gen:{}, reward:{}'.format(cur_gen, ind_result[0]))
        print('base_gauc_detail:{}'.format(base_gauc_detail))
        print('global_gauc_detail:{},{}'.format(g_best, g_best_result))
        if ind_result[0] > -1:
            print('******** success_gauc_detail:{}'.format(ind_result[1]))
        else:
            print('^^^^^^^^ fail_gauc_detail:{}'.format(ind_result[1]))
        print('position_detail:{}'.format(pop_x[i]))
        if ind_result[0] > p_best_result[i]:
            p_best[i] = pop_x[i]
            p_best_result[i] = ind_result[0]
        global g_best_result
        global g_best
        g_best_position_list.append(copy.deepcopy(pop_x[i]))
        g_best_result_list.append(copy.deepcopy(ind_result[0]))
        if ind_result[0] > g_best_result:
            g_best = copy.deepcopy(pop_x[i])
            g_best_result = copy.deepcopy(ind_result[0])
    end_time = time.time()
    print("one update take time:{}".format(end_time - begin_time))

def gaussion_process(df, is_coin_user, group_name):
    data_bc = sparkContext.broadcast(df)
    val_list = sparkContext.parallelize(range(partition_num)).map(
        lambda x: (x, x)
    ).partitionBy(partition_num).map(
        lambda x: x[1]
    )
    print("*************** sync data *******************")
    print('train data size:{}'.format(len(df)))
    print("*************** init start *******************")
    init_param(val_list, data_bc, is_coin_user)
    result_record = []
    for gen in range(NGEN):
        begin_time = time.time()
        print('_' * 50)
        print(gen)
        print('_' * 50)
        update_operator(val_list, data_bc, gen, is_coin_user)
        result_record.append([gen, g_best, g_best_result])
        end_time = time.time()
        print("one generation take time:{}".format(end_time - begin_time))
        print('best_position:{}'.format(g_best))
        print('best_result:{}'.format(g_best_result))
    g_best_result_history = np.array(g_best_result_list)
    g_best_position_history = np.array(g_best_position_list)
    print("---- End of (successful) Searching ----")
    print(g_best_position_history[g_best_result_history.argsort()[-1 * TRAIN_DATA_SIZE:]])
    print("history reward:{}".format(g_best_result_history))
    print("history index:{}".format(g_best_position_history))
    best_position_index = g_best_result_history.argsort()[-1 * TRAIN_DATA_SIZE:]
    print('best_position_index:{}'.format(best_position_index))
    g_best_distinct_result = []
    g_best_distinct_position_param = []
    g_best_distinct_position = []
    for position in best_position_index:
        cur_best_result = g_best_result_history[position]
        if cur_best_result not in g_best_distinct_result:
            g_best_distinct_result.append(cur_best_result)
            g_best_position_param_per = g_best_position_history[position]
            g_best_distinct_position_param.append(g_best_position_param_per)
            g_best_distinct_position_per = []
            for model in bound_x:
                g_best_distinct_position_per += g_best_position_param_per[model]
            g_best_distinct_position.append(g_best_distinct_position_per)
            print(cur_best_result)
            print(g_best_position_param_per)
            print(g_best_distinct_position_per)
    INIT_X = (g_best_distinct_position, g_best_distinct_position_param)
    print('init_x:{}'.format(INIT_X))
    Y = []
    X = []
    print('$' * 50)
    DETAIL = []
    for index, x in enumerate(INIT_X[1]):
        X.append(INIT_X[0][index])
        Y.append(g_best_distinct_result[index])
        DETAIL.append("train_data:" + str(index))
    print("*************** init finish *******************")
    print(X)
    print(Y)
    model = GaussianProcessRegressor()
    model.fit(X, Y)
    for i in range(GAUSSIAN_NGEN):
        x = opt_acquisition(X, model)
        actual = reward(val_list, data_bc, x[1], False, False, is_coin_user)
        print('gaussian gen:{}, reward:{}'.format(i, actual[0]))
        if actual[0] == -1:
            print('stage=%s, Result: x=%s' % (i, x[0]))
            print('Current Result detail=%s' % (actual[1]))
            print('Current position detail=%s' % (x[1]))
            continue
        X.append(x[0])
        Y.append(actual[0])
        DETAIL.append(actual[1])
        print('stage=%s, Result: x=%s' % (i, x[0]))
        print('Current Result detail=%s' % (actual[1]))
        print('Current position detail=%s' % (x[1]))
        model.fit(X, Y)
    ix = argmax(Y)
    print('%s, %s, Best Result: x=%s, y=%.3f' % (group_name, is_coin_user, X[ix], Y[ix]))
    print('%s, %s, Best Result detail=%s' % (group_name, is_coin_user, DETAIL[ix]))
    print('*' * 50)
    for index, x in enumerate(X):
        if Y[index] > 0:
            print('%s, all result: x=%s, y=%.5f' % (is_coin_user, x, Y[index]))
    print('*' * 50)

def cal_best_param(df, is_coin_user, group_name):
    global pop_x
    global pop_v
    global p_best
    global p_best_result
    global g_best_result
    global g_best
    global g_best_result_list
    global g_best_position_list
    global base_gauc_detail
    pop_x = []
    pop_v = []
    p_best = []
    p_best_result = [-1 for i in range(pop_size)]
    g_best_result = -1
    g_best = {}
    g_best_result_list = []
    g_best_position_list = []
    base_gauc_detail = ''
    print('.........describe.........')
    print(df.describe())
    gaussion_process(df, is_coin_user, group_name)


def merge_label(row):
    for label in ['share', 'commentshow', 'comment', 'follow', 'head']:
        if label == 'comment' and row[label + '_score'] >= 0.00149:
            return 1
        if label == 'commentshow' and row[label + '_score'] >= 0.0179:
            return 1
        if label == 'follow' and row[label + '_score'] >= 0.1426:
            return 1
        if label == 'head' and row[label + '_score'] >= 0.3036:
            return 1
        if label == 'share' and row[label + '_score'] >= 0.0048:
            return 1
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process spark scripts args")
    parser.add_argument("--input", type=str, required=False, help="hdfs input path",
                        default="hdfs://R2/projects/ci_rcmd/hdfs/prod/recommend/data_pipeline/feature_dump/video")
    parser.add_argument("--output", type=str, required=False, help="hdfs input path",
                        default="hdfs://R2/projects/ci_rcmd/hdfs/dev/video/sean.sun/pso/raw_data")
    print(bound_x)
    print(seek_model)
    args = parser.parse_args()
    input = args.input
    output = args.output

    hive = Hive()
    rows = hive.query(sql)
    df = pd.DataFrame(rows)
    df.columns = header
    df['is_interaction_user'] = df.apply(merge_label, axis=1)

    print('before filter:{}'.format(len(df)))
    df = df.groupby('user_id').filter(lambda x: (len(x) >= GAUC_MIN_DATA_SIZE) & (len(x) <= GAUC_MAX_DATA_SIZE))
    print('after filter:{}'.format(len(df)))

    cal_best_param(df, False, 'all user')
    cal_best_param(df[df['is_coin_user'] == 1], True, 'coin_user')
    cal_best_param(df[df['is_coin_user'] == 0], False, 'non_coin_user')

    hive.stop()



