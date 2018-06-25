# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from get_word_vector import get_vectors

def get_xgb(train, test, rs=None):
	vecs, clazz = train
	# モデルを学習
	X_train, X_test, Y_train, Y_test = train_test_split(vecs, clazz, test_size=0.1, random_state=rs)
	xgtrain = xgb.DMatrix(X_train, Y_train)
	xgvalid = xgb.DMatrix(X_test, Y_test)
	xgb_params = {
		'objective': 'multi:softmax',
		'num_class': 3,
		'eta': 0.01,
		'max_depth': 15,
		'max_leaves': 48,
		'silent': True,
		'random_state': rs
	}

	xgb_clf = xgb.train(
		xgb_params,
		xgtrain,
		30, 
		[(xgtrain,'train'), (xgvalid,'valid')],
		maximize=False,
		verbose_eval=10, 
		early_stopping_rounds=10
	)

	return xgb_clf, xgb.DMatrix(test[0])

if __name__ == '__main__':
	train, test = get_vectors()
	clf, xgb_test = get_xgb(train, test, rs=1)
	
	# クラス分類を行う
	vecs, clazz = test
	clz = clf.predict(xgb_test)
	report = classification_report(clazz, clz, target_names=['class1','class2','class3'])
	print(report)

