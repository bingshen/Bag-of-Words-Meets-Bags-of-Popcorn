# Bag-of-Words-Meets-Bags-of-Popcorn

参考代码：https://github.com/tjflexic/kaggle-word2vec-movie-reviews

题目链接：https://www.kaggle.com/c/word2vec-nlp-tutorial/data

成绩：
	1000维特征：
		word2vec+LR:0.88364
		doc2vec_dm+doc2vec_bow+LR:0.90336
		word2vec+doc2vec_dm+doc2vec_bow+LR:0.91392 //线性模型后，用LR再次线性融合
		word2vec+doc2vec_dm+doc2vec_bow+tf-idf+LR:0.89616 //直接把所有特征拼接起来，然后LR
	400维特征：
		word2vec+doc2vec_dm+doc2vec_bow+tf-idf+两层神经网络:0.90948

