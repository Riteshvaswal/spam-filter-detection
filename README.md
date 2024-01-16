# spam-filter-detection



import pandas as pd
import numpy as np
import numpy

df=pd.read_csv(r'/content/sample_data/train (1).csv')
df.shape

df.head()

df=df.dropna()
df.shape

df.shape

df.reset_index(drop=True,inplace=True)
df.head()

df = df.sample(frac = 1)

X=df.drop(['qid'],axis=1)
X.head()



df['target'].value_counts()

Y=df['target']
Y.head()

X.shape,Y.shape

import tensorflow as tf
tf.__version__

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

voc_size=10000

messages=X.copy()

messages.head()

messages['question_text'][1]

l1=[]
count=0

for i in messages['question_text']:
  count+=1
  print(count)
  print(i)
  print(len(i),i)
  l1.append(len(i))
max(l1)

sent_length=max(l1)
sent_length

messages.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):

    review = re.sub('[^a-zA-Z]', ' ', messages['question_text'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[1]

messages['question_text'][1]



onehot_repr=[one_hot(words,voc_size)for words in corpus]
onehot_repr

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

len(embedded_docs),Y.shape

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(Y)

X_final

X_final.shape,y_final.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

X_train.shape,y_train.shape

X_test.shape,y_test.shape

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=64)

# y_pred1=model.predict_classes(X_test)
y_pred1= model.predict(X_test)
y_pred1=np.round(y_pred1)
y_pred1=y_pred1.astype('float')

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred1))

## Creating model
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())

model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=64)

# y_pred2=model1.predict_classes(X_test)
y_pred2= model1.predict(X_test)
y_pred2=np.round(y_pred2)
y_pred2=y_pred2.astype('float')

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred2)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred2)

