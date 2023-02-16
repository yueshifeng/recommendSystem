# coding: utf-8

class Metrics:
    @staticmethod
    def binaryIntLabelAuc(preds, labels): # nlogn // 二分类auc
        assert(len(preds) == len(labels))
        pos, neg = sum(labels), len(labels) - sum(labels)
        if pos == len(labels) or pos == 0: return len(preds), 0.0
        pairs = sorted(zip(preds, labels), key=lambda p: p[0], reverse=True)

        inv, sum_inv = 0, 0
        for pred, label in pairs:
            if label == 1: inv += 1
            else:
                sum_inv += inv
        return round(sum_inv * 1.0 / pos / neg , 5)

    @staticmethod
    def floatLabelAuc(preds, labels): # nlogn // 一致性auc
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
        assert(len(preds) == len(labels))
        pairs = list(zip(preds, labels))
        rank = [values2 for values1, values2 in sorted(pairs, key=lambda x: x[0], reverse=True)]
        auc = inversePairsAucCal(rank)
        return auc

if __name__ == "__main__":
    labels, preds = [1, 0, 1], [0.1, 0.2, 0.14]
    print(Metrics.binaryIntLabelAuc(preds, labels))
    print(Metrics.floatLabelAuc(preds, labels))
