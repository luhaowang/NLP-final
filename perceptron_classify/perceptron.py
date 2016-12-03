import sys
import os
import random

class message_s:
    def __init__(self,style):
        self.style = style
        self.token_self_dict = {}

    def addFeatures(self, token):
        if token in self.token_self_dict:
            self.token_self_dict[token] += 1
        else:
            self.token_self_dict[token] = 1
    def changeStyle(self,style):
        self.style = style

#Initialization
document_path = sys.argv[1]
#out_put_file = sys.argv[2]

like_message_count = 0
dislike_message_count = 0

message_list = []
token_table = {}
bia_value = 0


#Prepare token table and message List

for root, dirs, files in os.walk(document_path):
    for file_name in files:
        if file_name.endswith(".txt"):
            lines = open(root+"/"+file_name, "r",encoding = "latin1").readlines()
            for each_lines in lines:
                tmp = each_lines.split("\n");
                token_arr = tmp[0].split(" ");
                new_message = message_s("DISLIKE")
                value = int(token_arr[0].split(".")[0])
                if value < 3:
                     dislike_message_count +=1
                else:
                     new_message.changeStyle("LIKE")
                     like_message_count +=1
                for i in range(1,len(token_arr)):
                    new_message.addFeatures(token_arr[i])
                    token_table[token_arr[i]] = 0
                message_list.append(new_message)


for i in range(0,20):
    random.shuffle(message_list)
    for j in range(0, len(message_list)):
        new_message = message_list[j]
        if new_message.style in "LIKE":
            label_value = 1
            alpha_value = 0
            new_dict_spam = new_message.token_self_dict
            for every_token in new_dict_spam:
                alpha_value += token_table[every_token]*new_dict_spam[every_token]
            alpha_value += bia_value
            if alpha_value*label_value <= 0:
                bia_value = bia_value + label_value
                for each_token in new_dict_spam:
                    token_table[each_token] = token_table[each_token] + label_value*new_dict_spam[each_token]
        elif new_message.style in "DISLIKE":
            label_value = -1
            alpha_value = 0
            new_dict_spam = new_message.token_self_dict
            for every_token in new_dict_spam:
                alpha_value += token_table[every_token]*new_dict_spam[every_token]
            alpha_value += bia_value
            if alpha_value*label_value <= 0:
                bia_value = bia_value + label_value
                for each_token in new_dict_spam:
                    token_table[each_token] = token_table[each_token] + label_value*new_dict_spam[each_token]


model_file_data = open("per_model.txt", "w",encoding = "latin1")


model_file_data.write("dislike_message_count %s" %dislike_message_count)
model_file_data.write("\n")
model_file_data.write("like_message_count %s" %like_message_count)
model_file_data.write("\n")
model_file_data.write("bia_value %s" %bia_value)
model_file_data.write("\n")

for token in token_table:
    model_file_data.write("%s %d" %(token, token_table[token]))
    model_file_data.write('\n')








