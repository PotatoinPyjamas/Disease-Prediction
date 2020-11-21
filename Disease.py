import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

test=pd.read_csv("test_data.csv",error_bad_lines=False)
train=pd.read_csv("training_data.csv",error_bad_lines=False)

test.head()
train.head()
train.info()
train=train.drop('Unnamed: 133',axis=1)
train.head()

y_train=train.prognosis
x_train=train.drop('prognosis',axis=1)
x_train
y_train

y_test=test.prognosis
x_test=test.drop('prognosis',axis=1)
x_test.head()
y_test.head()

f,ax = plt.subplots(figsize=(75,16))
sns.countplot(y_train,label="Count",ax=ax) 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)

pickle.dump(clr_rf,open('model.pkl','wb'))

col=x_train.columns
type(col)
len(col)

inputt = "itching stomach_pain skin_rash".split(' ')
inputt

b=[0]*len(col)
for x in range(0,132):
    for y in inputt:
        if(col[x]==y):
            b[x]=1
b=np.array(b)
b=b.reshape(1,132)
sol=clf_rf.predict(b)
sol

