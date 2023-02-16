# coding: utf-8
import random
import time

class Reader:
    def __init__(self, filename):
        self.filename = filename

    def parseLines(self, sample_rate=0.005):
        s = time.time()
        featureAndLabels = []
        for line in open(self.filename, "r"):
            if random.random() >= sample_rate: continue
            flag, featureAndLabel = self.parseLine(line)
            if not flag: continue
            featureAndLabels.append(featureAndLabel)
        print("Reader file: %s has rows: %s with sample_rate: %s, elapse %s s" % (self.filename, len(featureAndLabels), sample_rate, round(time.time() - s, 5)))
        return featureAndLabels

    def parseLine(self, line):
        lt = line.strip("\n").split("\t")
        if "\\N" in lt or "-1" in lt: return False, []
        user_id = int(lt[0])
        video_id = int(lt[1])
        bind_product_size = int(lt[2])

        ecom_anchor_clk_pred = float(lt[3])
        ecom_card_imp_pred = float(lt[4])
        ecom_card_clk_pred = float(lt[5])
        ecom_anchor_cvr_pred = float(lt[6])
        staytime_pred = float(lt[7])

        staytime_label = float(lt[8])
        video_anchor_click_label = 1 if int(lt[9]) > 0 else 0
        enhanced_card_impression_label = 1 if int(lt[10]) > 0 else 0
        enhanced_card_click_label = 1 if int(lt[11] ) > 0 else 0
        ecom_anchor_cvr_label = 1 if int(lt[12]) > 0 else 0
        return True, [staytime_pred, staytime_label, ecom_anchor_clk_pred, video_anchor_click_label, ecom_card_imp_pred * ecom_card_clk_pred, enhanced_card_click_label, ecom_anchor_cvr_pred, ecom_anchor_cvr_label]

if __name__ == "__main__":
    featureAndLabels = Reader("auc_spark2").parseLines()
