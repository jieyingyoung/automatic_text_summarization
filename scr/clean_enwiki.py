
import sys
import argparse

import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import codecs
from time import time

# one way of cleaning the texts
def remove_symbles(texts):
    symbles = '\<\>\.\,\《\》\[\]\「\」\}\:\：\;\、\；\'\·\(\)\（\）\{\，\。\!\,\;\:\?\"\"\/\-'
    texts_without_symbles =re.sub(r'[{}]+'.format(symbles),'',texts) 
    texts_without_digits = [re.sub(r'\d+','',line) for line in texts_without_symbles]
    return texts_without_digits

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-nu', '--file_number', type=str, nargs='?',
    #                     default='00',
    #                     help='the index of the file')

    # args = parser.parse_args()
    # file_number = args.file_number

    filenumbers = ['00','01','02','03','04','05','06','07','08','09','10'
                    ,'11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26']
    
    for file_number in filenumbers:

        begin = time()
        # open the English corpus using 'read' only 
        input_path = '../data/wiki_texts_{}_origin.txt'.format(file_number)
        f = codecs.open(input_path, 'r', encoding="utf8")
        # write to wiki_texts_number_tokens.txt
        output_path = '../data/wiki_texts_{}_tokens.txt'.format(file_number)
        output_file = codecs.open(output_path, 'w', encoding="utf8")

        line_num = 1
        line = f.readline()

        # Iterate each line, tokenize and remove symbles
        while line:
            print('---- processing ', line_num, ' article----------------')
            line_without_symples = "".join(remove_symbles(line))
            # print('text',line_without_symples)
            line_token = sent_tokenize(line_without_symples)
            # print('token',line_token)
            output_file.writelines(line_without_symples)
            line_num = line_num + 1
            line = f.readline()

        # close the file and exit
        f.close()
        output_file.close()
        end = time()
        print("wiki-{} cleaning time: {} seconds" .format(file_number,end - begin))
        # exit()
