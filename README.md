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

update:
	题目没有写提交的是类别还是概率，之前一直提交的是类别数据，始终无法突破0.92
	现在把提交改为了概率，然后提交。通过之前实验的情况，取最高精度的模型融合方法，得到了目前最好成绩：0.97004
	word2vec+doc2vec_dm+doc2vec_bow+LR：输出概率
	
获得最好成绩的运行步骤：
	1、python feature_w2v.py
	2、python feature_d2v.py
	3、python blending.py

其他的文件是一些单模型和其他的模型融合方法，效果并没有当前这个更优秀
