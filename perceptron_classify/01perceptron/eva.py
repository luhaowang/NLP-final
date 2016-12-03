import os
import sys

name = sys.argv[1]

result_data = open(name, "r", encoding = "latin1").readlines()
total_dislike = 0
total_like = 0
right_dislike = 0
right_like = 0
i_think_is_like = 0
i_think_is_dislike = 0

for each_lines in result_data:
    tmp = each_lines.split("\n")
    each_word = tmp[0].split();
    flag1 = int(each_word[0].split(":")[1])
    if flag1 == 2:
        i_think_is_like += 1
    elif flag1 == 1:
        i_think_is_dislike += 1
    score = int(each_word[len(each_word) - 1].split(".")[0])
    if score == 2:
        flag2 = "like"
    else:
        flag2 = "dislike"

    if flag2 == "like":
        total_like += 1
        if flag1 == 2:
            right_like += 1
    elif flag2 == "dislike":
        total_dislike += 1
        if flag1 == 1:
            right_dislike += 1

precision_like = right_like/i_think_is_like
precision_dislike = right_dislike/i_think_is_dislike
recall_like = right_like/total_like
recall_dilike = right_dislike/total_dislike

F1_like = (2*precision_like*recall_like)/(precision_like+recall_like)
F1_dilike = (2*precision_dislike*recall_dilike)/(precision_dislike + recall_dilike)

print("total_2" , total_like)
print("total_1", total_dislike)
print("right_2", right_like)
print("right_1", right_dislike)
print("i_think_is_2", i_think_is_like)
print("i_think_is_1",i_think_is_dislike)

print("precision_2 %s" %precision_like)
print("recall_2 %s" %recall_like)
print("F1_2 %s" %F1_like)

print("precision_1 %s" %precision_dislike)
print("recall_1 %s" %recall_dilike)
print("F1_1 %s" %F1_dilike)



