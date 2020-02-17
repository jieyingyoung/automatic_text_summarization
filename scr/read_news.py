import csv
import read_wiki as rw

def read_content(file):
    texts = ''
    with open(file,mode = 'r',encoding='gb18030',errors='ignore') as f_csv:
        csv_reader = csv.DictReader(f_csv)
        for row in csv_reader:
            # print(row['content'])
            texts += row['content']
    return texts

if __name__ == '__main__':
    file_path = 'C:/Users/psyji/Dropbox/data/kaikeba_nlp_project1/'
    file_name = 'sqlResult_1558435.csv'
    file = file_path+ file_name
    texts_origin = read_content(file)
    output_file = '../data/news_texts_origin.txt'
    rw.mkdir('../data')
    rw.save_content(output_file,texts_origin)

