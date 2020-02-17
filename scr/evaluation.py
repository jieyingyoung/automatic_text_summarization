from rouge_score import rouge_scorer
import os
from nltk.tokenize import sent_tokenize

def rouge_score(summary_origin:str, summary_produced:str):
	scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
	scores = scorer.score(summary_origin, summary_produced)
	# print(scores)
	return scores

def extract_title_and_content(article:str):
	sentences = []
	content = []
	for sent in article:
		if sent != '':
			# if sent[0].isdigit():
			#
			# 	sentences[-1][0] += sent_tokenize(sent)[0]
			# else:
			sentences.append(sent_tokenize(sent))
	title = "".join(sentences[0])
	for c in sentences[1:-1]:
		c = "".join(c)
		content.append(c)
	content = ".".join(content)
		# content = "".join(map(str,sentences[1:-1]))
	# print('title-------------\n', title)
	# print('content------------------\n', content)
	return title, content

def evaluation_results(evaluation_name : str, score):
	precision = score[evaluation_name][0]
	recall = score[evaluation_name][1]
	fmeasure = score[evaluation_name][2]
	return precision, recall, fmeasure

def print_stat(storehouse,item2,item3):
	import numpy as np
	print('--------------------------------------------------------------------------')
	print('{} recall',np.max(storehouse['recall']), np.min(storehouse['recall']), np.average(storehouse['recall']))
	print('{} precision', np.max(storehouse['precision']), np.min(storehouse['precision']),
		  np.average(storehouse['precision']))
	print('{} F1', np.max(storehouse['fmeasure']), np.min(storehouse['fmeasure']),
		  np.average(storehouse['fmeasure']))
	return

if __name__ == '__main__':
	topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
	topic = topics[0]
	evaluation1 = {'precision':[],'recall':[],'fmeasure':[]}
	evaluation2 = {'precision': [], 'recall': [], 'fmeasure': []}
	article_root='../evaluation_data/BBC_News_Summary/News_Articles/{}'.format(topic)
	summary_root = '../evaluation_data/BBC_News_Summary/Summaries/{}'.format(topic)
	article_lists=os.listdir(article_root)
	for s in article_lists:
		article_ = open(article_root + '/' + s, "r")
		article = article_.read().replace('\n', '.').split('.')
		title,content = extract_title_and_content(article)
		article_.close()

		summary_ = open(summary_root + '/' + s, "r")
		summary = summary_.read().split('.')
		summary = ". ".join(summary)
		# print('summary_origin-------------\n',summary)
		summary_.close()

		summary_produced = ' Quarterly profits at US media giant TimeWarner jumped 76% to $113bn (£600m) for the three months to December, '
		summary_origin = 'TimeWarner said fourth quarter sales rose 2% to $11. 1bn from $10. 9bn. For the full-year, TimeWarner posted a profit of $3. 36bn, '
		score = rouge_score(summary_origin,summary_produced)
		precision1, recall1, fmeasure1 = evaluation_results('rouge1',score)
		precision2, recall2, fmeasure2 = evaluation_results('rouge2', score)
		evaluation1['precision'].append(precision1)
		evaluation1['recall'].append(recall1)
		evaluation1['fmeasure'].append(fmeasure1)
		evaluation2['precision'].append(precision2)
		evaluation2['recall'].append(recall2)
		evaluation2['fmeasure'].append(fmeasure2)
	print(b)

