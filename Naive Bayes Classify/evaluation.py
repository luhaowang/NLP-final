import sys
import os
import math

count_actual_pos = 0
count_actual_neg = 0
count_correct_pos = 0
count_correct_neg = 0
count_my_pos = 0
count_my_neg = 0

my_file = open("nb_predict.txt", "r", encoding="utf-8").readlines()
ori_file = open("test_ori.txt", "r", encoding="utf-8").readlines()

for lines in range(0, len(my_file)):
    my_line = my_file[lines].split()
    ori_line = ori_file[lines].split()

    if my_line[0] == "POS":
        count_my_pos +=1
    if my_line[0] == "NEG":
        count_my_neg +=1
    if ori_line[0] == "POS":
        count_actual_pos +=1
    if ori_line[0] == "NEG":
        count_actual_neg +=1
    if my_line[0] == "POS" and ori_line[0] == "POS":
        count_correct_pos +=1
    if my_line[0] == "NEG" and ori_line[0] == "NEG":
        count_correct_neg +=1

print("actual_pos: ", count_actual_pos)
print("actual_neg: ", count_actual_neg)
print("correct_pos: ", count_correct_pos)
print("correct_neg: ", count_correct_neg)
print("my_pos: ", count_my_pos)
print("my_neg: ", count_my_neg)

accuracy = (count_correct_pos + count_correct_neg) / (len(my_file))
print("accuracy: ", accuracy)
precision_pos = count_correct_pos / count_my_pos
print("Precision(pos): ", precision_pos)
precision_neg = count_correct_neg / count_my_neg
print("Precision(neg): ", precision_neg)
recall_pos = count_correct_pos / count_actual_pos
print("Recall(pos): ", recall_pos)
recall_neg = count_correct_neg / count_actual_neg
print("Recall(neg): ", recall_neg)
F1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos)
print("F1 Score(pos): ", F1_pos)
F1_neg = (2* precision_neg * recall_neg) / (precision_neg + recall_neg)
print("F1 Score(neg): ", F1_neg)