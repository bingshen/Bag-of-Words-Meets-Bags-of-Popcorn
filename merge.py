import h5py
import numpy as np
import pandas as pd

if __name__ == '__main__':
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    with h5py.File("pred1.h5",'r') as h:
        pred1=np.array(h['pred'])
    with h5py.File("pred2.h5",'r') as h:
        pred2=np.array(h['pred'])
    pred=(pred1+pred2)/2.0
    sentiment=[]
    for x in pred:
        if x>=0.5:
            sentiment.append(1)
        else:
            sentiment.append(0)
    submission=pd.DataFrame({'id':test_df['id'],'sentiment':sentiment})
    submission.to_csv('submission.csv',index=False,quoting=3)