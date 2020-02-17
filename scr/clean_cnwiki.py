
import sys
import argparse

import json
import re
import jieba
from opencc import OpenCC
# import jieba.analyse
import codecs
from time import time

# one way of cleaning the texts
def remove_symbles(texts):
    symbles = '<>.,《》[]【】「」:：;、“”；·()（）{}《》，。!,;:?""'
    texts_without_symbles =re.sub(r'[{}]+'.format(symbles),'',texts)
    texts_without_digits = [re.sub(r'\d+','',line) for line in texts_without_symbles]
    texts_without_English = [re.sub("[a-zA-z]+","",line) for line in texts_without_digits]
    return texts_without_English

'''
# another way of cleaning the texts to take only the Chinese
def remove_symbles(texts):
    context = texts.decode("utf-8")  # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    context = filtrate.sub(r'', context)  # remove all non-Chinese characters
    context = context.encode("utf-8")  # convert unicode back to str
    return context
'''

#繁体转换为简体
def convert_chinese(string,config):
    # config: 's2t' 是简体转繁体，'t2s' 是繁体转简体
    cc = OpenCC(config)
    return cc.convert(string)

# to cut into tokens
def cut_texts(string):
    return list(jieba.cut(str(string).strip()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nu', '--file_number', type=str, nargs='?',
                        default='00',
                        help='the index of the file')
    parser.add_argument('-fd', '--folder_name', type=str, nargs='?',
                        default='AA',
                        help='the index of the file')
    
    args = parser.parse_args()
    file_number = args.file_number
    folder_name = args.folder_name

    begin = time()
    # 以读的方式打开原始的简体中文语料库
    input_path = '../data/wiki_texts_{}_origin.txt'.format(file_number)
    f = codecs.open(input_path, 'r', encoding="utf8")
    # 将分完词的语料写入到wiki_texts_test_tokens3.txt文件中
    output_path = '../data/wiki_texts_{}_tokens.txt'.format(file_number)
    output_file = codecs.open(output_path, 'w', encoding="utf8")

    line_num = 1
    line = f.readline()

    # 循环遍历每一行，并对这一行进行分词,繁体转简体和去除标点符号操作
    while line:
        print('---- processing ', line_num, ' article----------------')
        line_seg = " ".join(jieba.cut(line))
        line_simple = convert_chinese(str(line_seg), 't2s')
        line_without_symples = remove_symbles(line_simple)
        output_file.writelines(line_without_symples)
        line_num = line_num + 1
        line = f.readline()

    # 关闭两个文件流，并退出程序
    f.close()
    output_file.close()
    end = time()
    print("wiki-01 cleaning time: %d seconds" % (end - begin))
    exit()
