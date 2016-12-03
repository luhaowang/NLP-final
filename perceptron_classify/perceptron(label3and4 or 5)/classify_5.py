import sys
import os
import math

class dev_file:
    def __init__(self,name):
        self.name = name
        self.label = ""
        self.token_dict = {}
    def addfeatures(self, token):
        if token not in self.token_dict:
            self.token_dict[token] = 1
        else:
            self.token_dict[token] += 1

document_path = sys.argv[1]
output_filename = sys.argv[2]


like_data_dict = {}
dislike_data_dict = {}
test_data_dict ={}

train_data = open("per_model.txt", "r", encoding = "latin1").readlines()



prepare_like_data = train_data[1].split()
like_message_count = prepare_like_data[len(prepare_like_data) - 1]

prepare_dislike_data = train_data[0].split()
dislike_message_count = prepare_dislike_data[len(prepare_dislike_data) - 1]

prepare_bia_data = train_data[2].split()
bia_data = float(prepare_bia_data[len(prepare_bia_data) - 1])
token_weight_table = {}
result_table = {}
#complete training data
for i in range(4, len(train_data)):
    token_and_weight = train_data[i].split(" ")
    token_weight_table[token_and_weight[0]] = float(token_and_weight[len(token_and_weight) - 1])

#complete dev data
file_list = []
for root, dirs, files in os.walk(document_path):
    for file_name in files:
        if file_name.endswith(".txt"):
            lines = open(root+"/"+file_name, "r",encoding = "latin1").readlines()
            for eveLines in lines:
                tmp_arr = eveLines.split("\n")
                token_arr = tmp_arr[0].split(" ")
                new_file = dev_file(token_arr[0])
                value = int(token_arr[0].split(".")[0])
                if value >= 3:
                    for i in range(1,len(token_arr)):
                        new_file.addfeatures(token_arr[i])
                    file_list.append(new_file)

#calculation
for i in range (0, len(file_list)):
    dev_each_file = file_list[i]
    dev_table = dev_each_file.token_dict
    alpha_value  = 0
    for dis_token in dev_table:
        if dis_token in token_weight_table:
            alpha_value += dev_table[dis_token]*token_weight_table[dis_token]
    alpha_value += bia_data
    if alpha_value > 0:
        dev_each_file.label = "score:5"
    else:
        dev_each_file.label = "score:3or4"

#output
result_files = open(output_filename, "w",encoding = "latin1")
for i in range(0, len(file_list)):
    each_files = file_list[i]
    result_files.write("%s original: %s" %(each_files.label, each_files.name))
    result_files.write("\n")



