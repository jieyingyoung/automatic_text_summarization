import clean_wiki as cw
import codecs
import jieba


if __name__ == '__main__':
    input_path = '../data/news_texts_origin.txt'
    f = codecs.open(input_path, 'r', encoding="utf8")
    # 将分完词的语料写入到 tokens文件中
    output_path = '../data/news_texts_tokens.txt'
    output_file = codecs.open(output_path, 'w', encoding="utf8")

    line_num = 1
    line = f.readline()

    # 循环遍历每一行，并对这一行进行分词,繁体转简体和去除标点符号操作
    while line:
        print('---- processing ', line_num, ' article----------------')
        line_seg = " ".join(jieba.cut(line))
        line_simple = cw.convert_chinese(str(line_seg), 't2s')
        line_without_symples = cw.remove_symbles(line_simple)
        output_file.writelines(line_without_symples)
        line_num = line_num + 1
        line = f.readline()

    # 关闭两个文件流，并退出程序
    f.close()
    output_file.close()
    exit()






    #
    # texts_original = rsw.open_texts(file_path+input_file)
    # texts_pure = rsw.remove_symbles(texts_original)
    # tokens = rsw.cut_texts(texts_pure)
    # print(tokens)
    # output_file = '../data/news_texts_tokens.json'
    # rsw.save_content(output_file,tokens)


