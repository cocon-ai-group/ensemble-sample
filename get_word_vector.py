# -*- coding: utf-8 -*-
import sys
import codecs
import re
import os
import urllib.parse
import urllib.request
from janome.tokenizer import Tokenizer
from html.parser import HTMLParser
from sklearn.feature_extraction.text import TfidfVectorizer

# ダウンロードする記事
urls = [ \
	('共和政ローマ', '1.txt'), \
	('王政ローマ', '2.txt'), \
	('不思議の国のアリス', '3.txt'), \
	('ふしぎの国のアリス', '4.txt'), \
	('Python', '5.txt'), \
	('Ruby', '6.txt'), \
]

def get_files():
	# HTMLパーサー
	class MyParser(HTMLParser):
		def __init__(self, **args):
			self.inptag = False
			self.tagdata = []
			self.current = ''
			super(MyParser, self).__init__(**args)
			
		def handle_starttag(self, tag, attrs):
			if tag == 'p':
				self.inptag = True

		def handle_endtag(self, tag):
			if tag == 'p':
				self.inptag = False
				self.tagdata.append(self.current)
				self.current = ''

		def handle_data(self, data):
			if self.inptag:
				self.current = self.current + data

	# 形態素解析
	tk = Tokenizer()

	for url, dst in urls:
		# 日本語版Wikipediaのページ
		with urllib.request.urlopen('https://ja.wikipedia.org/wiki/'+ \
				urllib.parse.quote_plus(url)) as response:
			# URLから読み込む
			html = response.read().decode('utf-8')
		
			# 本文の<p>タグを取得する
			p = MyParser()
			p.feed(html)
		
			# 本文のみを取り出す
			with open(dst, 'w') as file:
				for a in p.tagdata:
					# 単語のリストにする
					l = [p.surface for p in tk.tokenize(a)]
					l = list(filter(lambda a: a.strip() != '', l))
					# 5単語以上
					if len(l) > 5:
						line = ' '.join(l)
						file.write(line)
						file.write('\n')

def get_vectors():
	if not (os.path.exists('1.txt') and os.path.exists('2.txt') and os.path.exists('3.txt') and
			os.path.exists('4.txt') and os.path.exists('5.txt') and os.path.exists('6.txt')):
		get_files()
	# 学習データを読み込む
	clz1txt = open('1.txt').readlines()
	clz2txt = open('3.txt').readlines()
	clz3txt = open('5.txt').readlines()

	# 全部繋げる
	alltxt = []
	alltxt.extend([s for s in clz1txt])
	alltxt.extend([s for s in clz2txt])
	alltxt.extend([s for s in clz3txt])

	# クラスを作成
	clazz_train = [0] * len(clz1txt) + [1] * len(clz2txt) + [2] * len(clz3txt)

	# TF-IDFベクトル化
	vectorizer = TfidfVectorizer(use_idf=True, token_pattern='(?u)\\b\\w+\\b')
	vecs_train = vectorizer.fit_transform(alltxt)
	
	# テスト用データを読み込む
	tst1txt = open('2.txt').readlines()
	tst2txt = open('4.txt').readlines()
	tst3txt = open('6.txt').readlines()
	alltxt = []
	alltxt.extend([s for s in tst1txt])
	alltxt.extend([s for s in tst2txt])
	alltxt.extend([s for s in tst3txt])

	# クラスを作成
	clazz_test = [0] * len(tst1txt) + [1] * len(tst2txt) + [2] * len(tst3txt)

	# TF-IDFベクトル化
	vecs_test = vectorizer.transform(alltxt)
	
	# 単語を保存
	with open('voc.txt', 'w') as f:
		f.write('\n'.join(vectorizer.vocabulary_.keys()))

	return ((vecs_train,clazz_train), (vecs_test,clazz_test))

	
	
if __name__ == '__main__':
	get_files()
