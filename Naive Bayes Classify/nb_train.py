import sys
import math
import os

class Cate:
    def __init__(self, name):
        self.name = name
        self.table = {}

    def addtoken(self, token, v):
        if token in self.table:
            self.table[token] += v
        else:
            self.table[token] = v

class Msg:
    def __init__(self, msg):
        self.msg = msg
        self.label = ""
        self.list = []

positive = Cate("POSITIVE")
negative = Cate("NEGATIVE")
pos_reviews = 0
neg_reviews = 0
pos_tokens = 0
neg_tokens = 0
voc_set = set()
pos_set = set()
neg_set = set()
pos_dict = {}
neg_dict = {}

train_lines = open("output.txt", "r", encoding = "utf-8").readlines()
for lines in train_lines:
    items = lines.split()
    # print(len(items))
    if(items[0]=="5.0" or items[0]=="4.0" or items[0]=="3.0"):
        pos_reviews +=1
        for i in range(1,len(items)):
            pos_tokens +=1
            voc_set.add(items[i])
            pos_set.add(items[i])
            positive.addtoken(items[i],1)
    if(items[0]=="2.0" or items[0]=="1.0" or items[0]=="0.0"):
        neg_reviews +=1
        for j in range(1,len(items)):
            neg_tokens +=1
            voc_set.add(items[j])
            neg_set.add(items[j])
            negative.addtoken(items[j],1)

# print(pos_reviews)
# print(neg_reviews)
# print(pos_tokens)
# print(neg_tokens)

for token, num in positive.table.items():
    pos_dict[token] = num
for token, num in negative.table.items():
    neg_dict[token] = num
for tokens in voc_set:
    if tokens not in pos_set:
        pos_dict[tokens] = 0
        pos_set.add(tokens)
    if tokens not in neg_set:
        neg_dict[tokens] = 0
        neg_set.add(tokens)

for all_items in pos_dict:
    pos_dict[all_items] +=1
for all_items in neg_dict:
    neg_dict[all_items] +=1

test_reviews = []
test_file = open("test_reviews.txt", "r", encoding="utf-8").readlines()
for test_lines in test_file:
    msg_classify = Msg(test_lines)
    lllist = test_lines.split()
    for ll in lllist:
        msg_classify.list.append(ll)
    test_reviews.append(msg_classify)

p_pos = math.log(pos_reviews/(pos_reviews+neg_reviews))
p_neg = math.log(neg_reviews/(pos_reviews+neg_reviews))

for review in test_reviews:
    p_pos_review = 0
    p_neg_review = 0
    for re in review.list:
        if re in pos_dict:
            p_pos_review += math.log((pos_dict[re]-1/10) / (pos_tokens + len(voc_set)*9/10))
        if re in neg_dict:
            p_neg_review += math.log((neg_dict[re]-1/10) / (neg_tokens + len(voc_set)*9/10))
    p_pos_final = p_pos + p_pos_review
    p_neg_final = p_neg + p_neg_review
    if p_pos_final > p_neg_final:
        review.label = "POS"
    elif p_pos_final < p_neg_final:
        review.label = "NEG"
    else:
        review.label = "NOT DECIDABLE"
    # print(review.label)

nbpredict = open("nb_predict.txt", "w", encoding="utf-8")
for result in test_reviews:
    nbpredict.write("%s %s" %(result.label, result.msg))