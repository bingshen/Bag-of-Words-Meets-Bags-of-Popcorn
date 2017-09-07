# Bag-of-Words-Meets-Bags-of-Popcorn

参考代码：https://github.com/tjflexic/kaggle-word2vec-movie-reviews

题目链接：https://www.kaggle.com/c/word2vec-nlp-tutorial/data

解法是用word2vec、doc2vec。然后用LR进行模型融合。目前成绩是0.91+不算特别理想，后期打算加入tf-idf进行训练

单模型情况：word2vec+LR单模型成绩为0.883+ doc2vec+LR单模型成绩0.90+
两个单模型结果再次线性组合（LR）可以得到0.91+