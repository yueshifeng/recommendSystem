#! /usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

b_lower = 1
b_upper = 10
b_upper_v2 = 0.001
b_fix = 1
c_lower = 1
c_lower_v2 = 500
#c_middle = 1
#c_middle_v2 = 5
c_upper = 20
c_upper_v2 = 1000
finish_upper_a = 15
finish_lower_a = 1
staytime_upper_a = 10
staytime_lower_a = 1
staytime_c = 10
skip_a = -12
interaction_upper_a = 10 
interaction_lower_a = 1
START_DAY = '2022-07-10'
END_DAY = '2022-07-10'
bound_x = {
    'finish': {
        "upper": [finish_upper_a, b_fix, c_upper], "lower": [finish_lower_a, b_fix, c_lower],
        "model_type": "finish",
        "param":[11.0036,1,8.5071], "coin_param":[12.4821,1.0,10.7172],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,
        "rank_name": "rank_finish_score", "base_param":[8.975974308766867, 0.001, 9.2304],
        "exp_param":[8.995771987166032,0.001,10.0], "min" : 1.0, "max" : 0.0
    },
    'staytime':{
        "upper": [staytime_upper_a, b_fix, staytime_c], "lower": [staytime_lower_a, b_fix, c_lower],
        "model_type": "staytime",
        "param":[7.3117,1,10], "coin_param":[3.1975,1.0,10],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,
        "rank_name" : "rank_complete_rate_score_v1", "base_param":[6.4076488598850405, 0.001, 8.8726],
        "exp_param":[6.209581496696553,0.001,7.8747], "min" : 1.0, "max" : 0.0
    },
    'skip':{
        "upper": [skip_a + 5, b_fix, c_upper], "lower": [skip_a - 5, b_fix, c_lower],
        "model_type": "skip",
        "param":[-8.551, 1, 8.1329],"coin_param":[-12.0919,1.0,5.6724],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,
        "rank_name" : "rank_skip_score",
        "base_param":[-12, 1, 10],
        "exp_param":[-2.8533589213069854,10.0,10.0], "min" : 1.0, "max" : 0.0
    },
    'like':{
        "upper": [interaction_upper_a, b_fix, c_upper], "lower": [interaction_lower_a, b_fix, c_lower],
        "model_type": "like",
        "param":[5.5916, 1, 14.8067],"coin_param":[6.0,1.0,9.3455],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,
        "rank_name": "rank_like_score", "base_param":[3.9788499449806016, 6.7762, 9.3992],
        "exp_param":[2.2652353393904865,10.0,6.787], "min" : 1.0, "max" : 0.0
    },
     'commentshow':{
         "upper": [interaction_upper_a, b_fix, c_upper], "lower": [interaction_lower_a, b_fix, c_lower],
         "model_type": "commentshow",
         "param":[5.6182, 1, 4.885],"coin_param":[6.0,1.0,4.12],
         "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,  "rank_name": "rank_comment_score", "base_param":[5.997446833974395, 3.166, 7.7446],
         "exp_param":[4.293082251341869,1.0,10.0], "min" : 1.0, "max" : 0.0
     },
    'share': {
        "upper": [interaction_upper_a, b_fix, c_upper_v2], "lower": [interaction_lower_a, b_fix, c_lower_v2],
        "model_type": "share",
        "param": [2.1347, 1, 940.9091],"coin_param":[2.6,1.0,926.7052],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,  "rank_name": "rank_complete_rate_score_v1",
        "base_param":  [3.6498306200403587, 1.0, 7.5498],
        "exp_param":[5.092506521934691,1.0,7.0306], "min" : 1.0, "max" : 0.0
    },
    'comment': {
        "upper": [interaction_upper_a, b_fix, c_upper_v2], "lower": [interaction_lower_a, b_fix, c_lower_v2],
        "model_type": "comment",
        "param": [2.4477, 1, 854.663],"coin_param":[3.0,1.0,771.6298],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,  "rank_name": "rank_skip_score",
        "base_param":  [3.9765560848794723, 1.0, 5.9686],
        "exp_param":[3.998120883184903,1.0,5.616], "min" : 1.0, "max" : 0.0
    },
    'follow': {
        "upper": [interaction_upper_a, b_fix, c_upper], "lower": [interaction_lower_a, b_fix, c_lower],
        "model_type": "follow",
        "param": [2.1044, 1, 9.4131],"coin_param":[3.1968,1.0,9.6284],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,  "rank_name": "rank_like_score",
        "base_param":  [0.9973304787518741, 6.7531, 6.3226],
        "exp_param":[0.9995302207962258,10.0,8.1479], "min" : 1.0, "max" : 0.0
    },
    'head': {
        "upper": [interaction_upper_a, b_fix, c_upper], "lower": [interaction_lower_a, b_fix, c_lower],
        "model_type": "head",
        "param": [2.3391, 1, 11.6726],"coin_param":[2.3816,1.0,8.6762],
        "gauc" : 0.0, "gauc_0" : 0.0, "gauc_1" : 0.0,  "rank_name": "rank_comment_score",
        "base_param":  [0.9973304787518741, 7.6087, 5.2457],
        "exp_param":[0.9995302207962258,5.6698,8.8228], "min" : 1.0, "max" : 0.0
    },
}
base_1_vv_ratio = 1.0
base_3_vv_ratio = 1.0
seek_model = list(bound_x.keys())
seek_model.sort()
init_seed_cnt = 100
train_seed_cnt = 1000
pop_size = 100
log_info = 10
NGEN = 10
GAUSSIAN_NGEN = 200
TRAIN_DATA_SIZE = 500
TOP_20_VIDEO = 20
TRAIN_DATA_IGNORE_SUPPRESS = False
FINISH_LIKE_COMMENT_FIX = False
GAUC_MIN_DATA_SIZE = 20
GAUC_MAX_DATA_SIZE = 200
c1 = 2  # 学习因子，一般为2
c2 = 2
w_init = 0.4
w_final = 0.2
cold_model_config =["in1day","in3day"]
rate = 0.9

sql = '''
            with post (select video_id as vid,  min(cast(ptime as bigint)/1000) as ptime, min(video_source) as video_source
            from video.video_mart_dim_video group  by video_id)
            select 
                element_at(extra_map, 'video_id_rank_interact_multihead_comment_pred') as comment_score,
                case when `comment` is not null and `comment` > 0 then 1 else 0 end as comment_label,
                element_at(extra_map, 'video_id_rank_interact_multihead_click_comment_pred') as commentshow_score,
                case when element_at(extra_act_map, 'click_comment') is not null and element_at(extra_act_map, 'click_comment') > 0 then 1 else 0 end as commentshow_label,
                element_at(extra_map, 'video_id_rk_finish_deepfm_h_v3') as finish_score,
                case when watch_duration > 0.9 * video_duration then 1 else 0 end as finish_label,
                element_at(extra_map, 'video_rank_follow_v2') as follow_score,
                case when follow > 0 then 1 else 0 end as follow_label,
                element_at(extra_map, 'video_rank_head_v1') as head_score,
                case when element_at(extra_act_map, 'click_avatar') is not null and element_at(extra_act_map, 'click_avatar') > 0 then 1 else 0 end as head_label,
                element_at(extra_map, 'video_id_rank_interact_multihead_like_pred') as like_score,
                case when `like` > 0 then 1 else 0 end as like_label,
                element_at(extra_map, 'video_id_rank_interact_multihead_share_pred') as share_score,
                case when share > 0 then 1 else 0 end as share_label,
                element_at(extra_map, 'video_id_rk_autoint_skip2') as skip_score,
                case when watch_duration < 3000 then 1 else 0 end as skip_label,
                element_at(extra_map, 'video_id_rk_staytime_mc_mse_expectation_pred') as staytime_score,
                watch_duration / 1000 as staytime_label,
                element_at(extra_map, 'final_rerank_score') / (element_at(extra_map, 'video_multi_adjust_score') + 1.0) as boost_score,
                user_id,
                hash(user_id) % 600 as bucket,
                if(element_at(extra_info_int_map, 'gold_coin_days') >= 3, 1, 0) as is_coin_user,
                element_at(extra_info_int_map, 'is_newbee') as is_newbee
                --,case when req_time - post.ptime <= 3600 * 24 then 1 else 0 end as in1dayvv,
                --case when req_time - post.ptime <= 3600 * 24 * 3 then 1 else 0 end as in3dayvv
            from ci_rcmd.video_feature_dump fd left join post on post.vid = fd.item_id
            where grass_date between '{}' and '{}' and watch_duration > 0 and video_duration > 0
                and element_at(extra_map, 'video_multi_adjust_score') is not null 
                and element_at(extra_map, 'final_rerank_score') is not null
                and element_at(extra_map, 'video_id_rk_finish_deepfm_h_v3') is not null 
                and element_at(extra_map, 'video_id_rk_staytime_mc_mse_expectation_pred') is not null
                and element_at(extra_map, 'video_id_rk_autoint_skip2') is not null 
                and element_at(extra_map, 'video_id_rank_interact_multihead_like_pred') is not null
                and element_at(extra_map, 'video_id_rank_interact_multihead_comment_pred') is not null
                and element_at(extra_map, 'video_id_rank_interact_multihead_share_pred') is not null
                and element_at(extra_map, 'video_rank_follow_v2') is not null
                and element_at(extra_map, 'video_rank_head_v1') is not null
                and element_at(extra_map, 'video_id_rank_interact_multihead_click_comment_pred') is not null
                and element_at(extra_map, 'final_rerank_score') > 0.0
                and element_at(extra_map, 'video_multi_adjust_score') > 0.0
                and element_at(extra_map, 'final_rerank_score') / element_at(extra_map, 'video_multi_adjust_score') + 1 > element_at(extra_map, 'final_rerank_score') / element_at(extra_map, 'video_multi_adjust_score')
                and element_at(extra_map, 'final_rerank_score') / (element_at(extra_map, 'video_multi_adjust_score')) > 0.1
                and element_at(extra_map, 'final_rerank_score') / (element_at(extra_map, 'video_multi_adjust_score')) < 5000 
                and element_at(extra_info_int_map, 'is_newbee') == 0
          '''.format(START_DAY, END_DAY)

header = []
for model in seek_model:
    print(bound_x[model]["model_type"])
    header += [bound_x[model]["model_type"] + "_score", bound_x[model]["model_type"] + "_label"]
header.append("boost_score")
header.append("user_id")
header.append("bucket")
header.append("is_coin_user")
header.append("final_rank_score")
# header.append("in1day")
# header.append("in3day")

def cal_mixed_score(ind_var, df):
    mixed_score = 1.0
    score_record = {}
    raw_score_record = {}
    sum_a = 0
    for model_name in bound_x:
        sum_a += ind_var[model_name][0]
    #mixed_score *= np.array(df["boost_score"])
    mixed_score *= 1.0
    for model_name in bound_x:
        c_item = 1.0
        # if bound_x[model_name]["model_type"] == "staytime":
        #     c_item = /1000
        a = ind_var[model_name][0]
        b = ind_var[model_name][1]
        c = ind_var[model_name][2]
        score = np.array(df[bound_x[model_name]["model_type"] + "_score"])
        # if bound_x[model_name]["model_type"] == "staytime":
        #     score = np.power(2, (score / (1 - score)) - 1) - 1
        raw_score_record[model_name] = score
        sub_score = np.power((b + c * score), a)
        sub_score /= np.power(10.0, a)
        score_record[model_name] = sub_score
        mixed_score *= sub_score
    return mixed_score


def cal_mixed_score_test(ind_var, df):
    mixed_score = 1.0
    score_record = {}
    raw_score_record = {}
    sum_a = 0
    for model_name in bound_x:
        sum_a += ind_var[model_name][0]
    mixed_score *= np.array(df["boost_score"])
    for model_name in bound_x:
        c_item = 1.0
        # if bound_x[model_name]["model_type"] == "staytime":
        #     c_item = /1000
        a = ind_var[model_name][0]
        b = ind_var[model_name][1]
        c = ind_var[model_name][2]
        score = np.array(df[bound_x[model_name]["model_type"] + "_score"])
        #if bound_x[model_name]["model_type"] == "staytime":
        #    score = np.power(2, (score / (1 - score)) - 1) - 1
        print(model_name)
        df = pd.DataFrame(score)
        print(df.describe())
        raw_score_record[model_name] = score
        sub_score = np.power((b + c * score), a)
        sub_score /= np.power(10.0, a)
        score_record[model_name] = sub_score
        mixed_score *= sub_score
    return mixed_score

def cal_group_auc(labels, preds, user_id_list, is_spearmanr):
    """Calculate group auc"""
    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag
    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            if is_spearmanr:
                auc = calAUC(np.asarray(group_score[user_id]), np.asarray(group_truth[user_id]))
            else:
                auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    if impression_total == 0:
        return (0, 0)
    return (float(total_auc), impression_total)

def cal_group_ratio(labels, preds, user_id_list):
    """Calculate group auc"""
    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)
    cold_start_video_cnt = 0
    for user_id in set(user_id_list):
        scores = group_score[user_id]
        cold_start_video_cnt += sum(np.array(group_truth[user_id])[np.array(scores).argsort()[-1 * TOP_20_VIDEO:]])
    return cold_start_video_cnt

def calAUC(prob, labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0], reverse=True)]
    auc = inversePairsAucCal(rank)
    return auc

def inversePairsAucCal(data):
    n = len(data)
    def merge(left, right):
        count = 0
        l = r = 0
        result = []
        while l < len(left) and r < len(right):
            if left[l] <= right[r]:
                result.append(left[l])
                l += 1
            else:
                result.append(right[r])
                r += 1
                count += len(left) - l
        result += left[l:] + right[r:]
        return count, result
    def merge_sort(a_list):
        l = 0
        r = len(a_list)
        mid = (l + r) // 2
        count = 0
        if len(a_list) <= 1:
            return count, a_list
        # 拆分
        count_l, left = merge_sort(a_list[:mid])
        count_r, right = merge_sort(a_list[mid:])
        # 合并排序
        count_merge, mergeData = merge(left, right)
        count = count_l + count_r + count_merge
        return count, mergeData
    count, result = merge_sort(data)
    return float(count) / (n * (n - 1) / 2)

def gauc_worker(df, model_type, mixed_score, user_id_list):
    flag = True if model_type == 'staytime' else False
    labels = np.array(df[model_type + "_label"])
    return cal_group_auc(labels, mixed_score, user_id_list, flag)

def cold_video_worker(df, model_type, mixed_score, user_id_list):
    labels = np.array(df[model_type])
    return cal_group_ratio(labels, mixed_score, user_id_list)

def cal_reward(params, index, data_bc):
    df_copy = data_bc[abs(data_bc['bucket']) == index].copy()
    mixed_score = cal_mixed_score(params, df_copy)
    # mixed_score = np.array(df_copy["final_rank_score"])
    user_id_list = np.array(df_copy["user_id"])
    model_gauc = {}
    for model in seek_model:
        if bound_x[model]["model_type"] not in ('finish', 'staytime', 'commentshow', 'like', 'follow', 'comment', 'share', 'head'):
        #if bound_x[model]["model_type"] not in ('finish', 'staytime'):
            continue
        model_gauc[bound_x[model]["model_type"]] = gauc_worker(df_copy, bound_x[model]["model_type"], mixed_score, user_id_list)
    # for cold_model in cold_model_config:
    #     model_gauc[cold_model] = cold_video_worker(df_copy, cold_model, mixed_score, user_id_list)
    return model_gauc

def cal_reward_test(params, index, data_bc):
    df_copy = data_bc[abs(data_bc['bucket']) == index].copy()
    mixed_score = cal_mixed_score_test(params, df_copy)
    model_gauc = {}
    return model_gauc

def get_model_name(key):
    for model in bound_x:
        if bound_x[model]['model_type'] == key:
            return model
    return None

def reward_v2(val_list, df, params, switch, mark):
    reward = 0
    gauc_detail = ''
    for i in range(0,2):
        df_table = df.value
        base_key = 'gauc_' + str(i)
        df_part = df_table[df_table['duration'] == i]
        model_gauc_list = val_list.map(
            lambda x: cal_reward(params, x, df_part)
        ).collect()
        gauc_numerator = {} # 分子
        gauc_denominator = {} # 分母
        day_1_cold_video_cnt = 0
        day_3_cold_video_cnt = 0
        for model_gauc in model_gauc_list:
            for key, value in model_gauc.items():
                if key in cold_model_config:
                    if key == 'in1day':
                        day_1_cold_video_cnt += value
                    if key == 'in3day':
                        day_3_cold_video_cnt += value
                    continue
                gauc_numerator[key] = gauc_numerator.get(key, 0) + value[0]
                gauc_denominator[key] = gauc_denominator.get(key, 0) + value[1]
        for key in gauc_numerator:
            model_name = get_model_name(key)
            if gauc_denominator[key] == 0:
                print("key={}, is zero={} ".format(key, gauc_denominator[key]))
                continue
            gauc_per_model = gauc_numerator[key] * 1.0000 / gauc_denominator[key]
            gauc_per_model = (1 - gauc_per_model) if key == 'skip' else gauc_per_model
            # if key == 'skip' and gauc_per_model < 0.5 and not switch:
            #     return (-1, "skip not valid:" + str(gauc_per_model) + ":" + str(bound_x[model_name]['gauc']))
            #if key == 'staytime' and gauc_per_model < bound_x[model_name]['gauc']  and not switch:
            #      return (-1, "staytime not valid:" + str(gauc_per_model)+ ":" + str(bound_x[model_name]['gauc']))
            #if (key == 'like' or key == 'share' or key == 'follow' or key == 'comment') and gauc_per_model < bound_x[model_name]['gauc'] - 0.1  and not switch:


            if (key == 'share' or key == 'commentshow' or key == 'comment') and gauc_per_model < bound_x[model_name][base_key] - 0.01  and not switch:
                return (-1, "part" + str(i) + ":" + str(key) + " not valid:" + str(gauc_per_model)+ ":" + str(gauc_per_model - bound_x[model_name][base_key]))
            if (key == 'like' or key == 'follow' or key == 'head') and gauc_per_model < bound_x[model_name]['gauc'] - 0.01  and not switch:
                return (-1, "part" + str(i) + ":" + str(key) + " not valid:" + str(gauc_per_model)+ ":" + str(gauc_per_model - bound_x[model_name][base_key]))
            if (key == 'finish' or key == 'staytime') and gauc_per_model < bound_x[model_name][base_key]:
                return (-1, "part" + str(i) + ":" + str(key) + " not valid:" + str(gauc_per_model)+ ":" + str(gauc_per_model - bound_x[model_name][base_key]))

           # if key == 'staytime':
           #     reward += gauc_per_model
            #     gauc_detail += key + ':' + str(gauc_per_model) + ','
            # elif key == 'finish':
            #     reward += gauc_per_model
            #     gauc_detail += key + ':' + str(gauc_per_model) + ','
            # elif key == 'skip':
            #     reward += gauc_per_model
            #     gauc_detail += key + ':' + str(gauc_per_model) + ','
            # else:
            #     gauc_detail += key + ':' + str(gauc_per_model) + ','
            if mark:
                bound_x[model_name][base_key] = gauc_per_model
            tmp_reward = gauc_per_model - bound_x[model_name][base_key]
            if key == 'finish' and i == 1:
                reward += 100 * tmp_reward
                print("boost:{},{}".format(key, tmp_reward))
            elif key == 'staytime' and i == 0:
                reward += 100 * tmp_reward
                print("boost:{},{}".format(key, tmp_reward))
            else:
                reward += tmp_reward
            gauc_detail += str(i) + ':' + key + ':' + str(gauc_per_model) + ' diff: ' + str(gauc_per_model - bound_x[model_name][base_key]) + ','
    gauc_detail += 'reward:' + str(reward)
    if mark:
        global base_gauc_detail
        print("base:{}".format(gauc_detail))
        base_gauc_detail = gauc_detail
    else:
        print("random params:{}, gauc:{}".format(params, gauc_detail))
    return (reward, gauc_detail)

def reward(val_list, df, params, switch, mark, is_coin_user):
    # cal_reward_test(params,  1, df.value)
    model_gauc_list = val_list.map(
        lambda x: cal_reward(params, x, df.value)
    ).collect()
    gauc_numerator = {} # 分子
    gauc_denominator = {} # 分母
    day_1_cold_video_cnt = 0
    day_3_cold_video_cnt = 0
    for model_gauc in model_gauc_list:
        for key, value in model_gauc.items():
            if key in cold_model_config:
                if key == 'in1day':
                    day_1_cold_video_cnt += value
                if key == 'in3day':
                    day_3_cold_video_cnt += value
                continue
            gauc_numerator[key] = gauc_numerator.get(key, 0) + value[0]
            gauc_denominator[key] = gauc_denominator.get(key, 0) + value[1]
    print("cold_video_cnt:{},{}".format(day_1_cold_video_cnt, day_3_cold_video_cnt))
    gauc_detail = ''
    reward = 0
    for key in gauc_numerator:
        model_name = get_model_name(key)
        if gauc_denominator[key] == 0:
            print("key={}, is zero={} ".format(key, gauc_denominator[key]))
            continue
        gauc_per_model = gauc_numerator[key] * 1.0000 / gauc_denominator[key]
        gauc_per_model = (1 - gauc_per_model) if key == 'skip' else gauc_per_model
        if mark:
            bound_x[model_name]['gauc'] = gauc_per_model
        tmp_reward = gauc_per_model - bound_x[model_name]['gauc']
        # tmp_reward = gauc_per_model
        if is_coin_user:
            if (key == 'finish' or key == 'staytime') and gauc_per_model < bound_x[model_name]['gauc'] and not switch:
                return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            if (key == 'commentshow' or key == 'head') and gauc_per_model < bound_x[model_name]['gauc'] and not switch:
                return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            # if (key == 'share' or key == 'commentshow' or key == 'comment' or key == 'like' or key == 'follow' or key == 'head') \
            #         and gauc_per_model < bound_x[model_name]['gauc'] - 0.01 and not switch:
            #     return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            if (key == 'share' or key == 'comment' or key == 'follow' or key == 'like') \
                    and gauc_per_model < bound_x[model_name]['gauc'] - 0.1 and not switch:
                return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            if key == 'staytime' or key == 'commentshow':
                tmp_reward = 100 * tmp_reward
            if key == 'finish' or key == 'head':
                tmp_reward = 10 * tmp_reward
            reward += tmp_reward
        else:
            if (key == 'finish' or key == 'staytime') and gauc_per_model < bound_x[model_name]['gauc'] and not switch:
                return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            if (key == 'commentshow' or key == 'like') and gauc_per_model < bound_x[model_name]['gauc'] and not switch:
                return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            # if (key == 'share' or key == 'commentshow' or key == 'comment' or key == 'like' or key == 'follow' or key == 'head') \
            #         and gauc_per_model < bound_x[model_name]['gauc'] - 0.01 and not switch:
            #     return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            if (key == 'share' or key == 'comment' or key == 'follow' or key == 'head') \
                    and gauc_per_model < bound_x[model_name]['gauc'] - 0.1 and not switch:
                return (-1, str(key) + " not valid:" + str(gauc_per_model) + ":" + str(gauc_per_model - bound_x[model_name]['gauc']))
            if key == 'staytime' or key == 'finish':
                tmp_reward = 100 * tmp_reward
            if key == 'commentshow' or key == 'like':
                tmp_reward = 10 * tmp_reward
            reward += tmp_reward
        gauc_detail += key + ':' + str(gauc_per_model) + 'diff:' + str(gauc_per_model - bound_x[model_name]['gauc']) + ','
    gauc_detail += 'reward:' + str(reward)
    if mark:
        global base_gauc_detail
        print("base:{}".format(gauc_detail))
        base_gauc_detail = gauc_detail
    else:
        print("random params:{}, gauc:{}".format(params, gauc_detail))
    return (reward, gauc_detail)
