import os
import sys
import random
import json
from pprint import pprint
from html.parser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ("Encountered a start tag:", tag)

document_path = sys.argv[1]

output_data = open("test.txt", "w",encoding = "utf-8")

max_score = 0
min_score = 0
count = 0

for root, dirs, files in os.walk(document_path):
    for file_name in files:
        if file_name.endswith(".json"):
            lines = open(root + "/" +file_name, "r").readlines()
            for key in range(0, len(lines)):
                data = json.loads(lines[key])
                #html_parser = MyHTMLParser()
                parser = MyHTMLParser()
                text = parser.unescape(data["reviewText"])
                output_data.write("%s %s" %(data["overall"], text))
                count += 1
                output_data.write('\n')

print(count)

test_reviews = open("test_reviews.txt", "w", encoding="utf-8")
ori_reviews = open("test_ori.txt", "w", encoding="utf-8")
files = open("test.txt", "r", encoding="utf-8").readlines()
for lines in files:
    line = lines.split()
    test_reviews.write(lines[4:])
    if(line[0]=="5.0" or line[0]=="4.0" or line[0]=="3.0"):
        ori_reviews.write("POS " + lines[4:])
    if(line[0]=="2.0" or line[0]=="1.0" or line[0]=="0.0"):
        ori_reviews.write("NEG " + lines[4:])
test_reviews.write('\n')
ori_reviews.write('\n')