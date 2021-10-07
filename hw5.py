# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 04:30:07 2020

@author: user
"""

import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,precision_recall_fscore_support

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.feature_extraction.text import CountVectorizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec,word2vec

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt  
"""

#資料前處理-刪除HTML tag
import re 
#匯入Regular Expression模組
#為了移除HTML tag

def rm_tags(text): 
    re_tag = re.compile(r'<[^>]+>') 
#建立re_tag正規表示式變數為'<[^>]+>'
    return re_tag.sub(' ',text) 
#使用re_tag將text文字中符合正規表示式的字替換成空字串

"""

def show_train_acc_history(train, val):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    

def show_train_loss_history(train, val):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()  


test = pd.read_csv("test.csv", delimiter="\t")#,error_bad_lines=False
train = pd.read_csv("train.csv", delimiter="\t")#,error_bad_lines=False
sub = pd.read_csv("sample_submission.csv")

#path = r'C:\Users\Michel Spiero\Desktop\Analise Python Optitex\Analytics Optitex\base_entrada_python_v2.csv'

# open(path, 'r', encoding='utf-8') as f:
#    entrada_arquivo = pd.read_csv(f, sep=';|"', engine='python')\
#                                               .dropna(how='all', axis=1)
train= train.drop([1615])
value_list = [row[0] for row in train.itertuples(index=False, name=None)]#tainx
label = [row[1] for row in train.itertuples(index=False, name=None)]#trainy
ans = [row[1] for row in sub.itertuples(index=False, name=None)]#testy
test_list = [row[1] for row in test.itertuples(index=False, name=None)]#testx


"""
for n in label:
    if n != '1' and n != '0':
        n = '0'
label[1615]='0';
"""
#for a in label:
#    a = int(a)
new_numbers = []
for n in label:
   new_numbers.append(int(n))
label = new_numbers


#label = [ int(x) for x in label ]

# Python2.x,可以使用map函数
# numbers = map(int, numbers) 

# python3.x,map返回的是map对象，也可以转换为List
#numbers = list(map(int, label)) 


import nltk
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')

import spacy

nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
"""
#stop_words=['the', 'a','and','an','he','she','her','him','his']
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None,stop_words=nltk_stopwords)#(max_df=0.95, min_df=2,
#統計每個詞語的tf-idf權值
filtered_word_csr = cv.fit_transform(value_list)
filtered_word_list = filtered_word_csr.A.tolist()


cv2 = CountVectorizer(max_df=1.0, min_df=1, max_features=None,stop_words=nltk_stopwords)
filtered_test_csr = cv2.transform(test_list)
filtered_test_list  = filtered_test_csr.A.tolist()
"""


filtered_word_list = value_list[:] #make a copy of the word_list
for i in range(len(filtered_word_list)): # iterate over word_list
    filtered_word_list[i] = [word for word in filtered_word_list[i].split() if word not in nltk_stopwords]

filtered_test_list = test_list[:] #make a copy of the word_list
for i in range(len(filtered_test_list)): # iterate over word_list   
    filtered_test_list[i] = [word for word in filtered_test_list[i].split() if word not in nltk_stopwords]

"""
        if word  not in nltk_stopwords: 
            filtered_sentence.append(word) # remove word from filtered_word_list if it is a stopword
            sentence = filtered_sentence
"""
"""
transformer = TfidfTransformer()
#第一個fit_transform是計算tf-idf，第二個fit_transform是將文本轉詞本矩陣
tfidf = transformer.fit_transform(cv.fit_transform(value_list))  
"""
"""
#TfidfVectorizer :Equivalent to CountVectorizer followed by TfidfTransformer.
vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=None,stop_words=nltk_stopwords)

X = vectorizer.fit_transform(value_list)
Y = vectorizer.transform(test_list)#['text']#	Transform documents to document-term matrix
"""
#cv.fit_transform(value_list)
"""
weightlist = X.toarray()  
wordlist = vectorizer.get_feature_names()
#print每類文本的tf-idf詞語權重，第一個for遍歷所有文本，第二個for遍歷某一類文本下的詞語權重
for i in range(len(weightlist)):  
    print ("-------這裡输出第",i,"類文本的詞語tf-idf權重------"  )
    for j in range(len(wordlist)):  
        print (wordlist[j],weightlist[i][j])
"""        
print("\n\n\n")
#print(vectorizer.get_feature_names())
#print(X.shape)


model = word2vec.Word2Vec(sentences=common_texts, window=5, min_count=1, workers=4)#vector_size=100,
#model = word2vec.Word2Vec(sentences, size=5, min_count=1, negative=10)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

#x_train = cv.fit_transform(value_list)
#To train Word2Vec it is better not to remove stop words because the algorithm relies on the broader context of the sentence in order to produce high-quality word vectors
model.train(value_list, total_examples=1, epochs=1)

vector = model.wv['computer']
word_vectors = model.wv






#建立Token
token = Tokenizer(num_words=3800) 
#使用Tokenizer模組建立token，建立一個3800字的字典
token.fit_on_texts(filtered_word_list)  
#讀取所有訓練資料，依照每個英文字在訓練資料出現的次數進行排序，
#前3800名的英文單字會加進字典中
token.word_index
#可以看到它將英文字轉為數字的結果，例如:the轉換成1

x_train_seq = token.texts_to_sequences(filtered_word_list)
x_test_seq = token.texts_to_sequences(filtered_test_list)
#透過texts_to_sequences可以將訓練和測試集資料中的文字轉換為數字list

x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test = sequence.pad_sequences(x_test_seq, maxlen=380)
#長度小於380的，前面的數字補0
#長度大於380的，截去前面的數字
#變成25000*380的矩陣 = 25000則，每則包含380個數字

transformer = TfidfTransformer()
#第一個fit_transform是計算tf-idf，第二個fit_transform是將文本轉詞本矩陣
x_traint = transformer.fit_transform(x_train) 
x_testt  = transformer.transform(x_test) 



#layer = tf.keras.layers.Dropout(.2, input_shape=(380,))
#outputs = layer(x_train.astype(np.float32), training=True)

#匯入模組
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

#建立Keras的Sequential模型
modelRNN = Sequential()  #建立模型
#Embedding層將「數字list」轉換成「向量list」
modelRNN.add(Embedding(output_dim=32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
     input_dim=3800,  #輸入的維度是3800，也就是我們之前建立的字典是3800字
     input_length=380)) #數字list截長補短後都是380個數字

#加入Dropout，避免overfitting
modelRNN.add(Dropout(0.7)) 	#隨機在神經網路中放棄70%的神經元，避免overfitting

#建立RNN層
modelRNN.add(SimpleRNN(units=16))
 #建立16個神經元的RNN層

#建立隱藏層
modelRNN.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
#ReLU激活函數
modelRNN.add(Dropout(0.7))#0.35

#建立輸出層
modelRNN.add(Dense(units=1,activation='sigmoid'))
#建立一個神經元的輸出層
#Sigmoid激活函數

#查看模型摘要
modelRNN.summary()

#定義訓練模型
modelRNN.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 
#import torch   pip uninstall torch
#train_history = modelRNN.fit(outputs, torch.Tensor(label), 
#xxx = outputs.numpy()#outputs.numpy().astype(np.int32)
train_history = modelRNN.fit(x_train, np.array(label), 
         epochs=10,#10 
         batch_size=100,
         verbose=2,
         validation_split=0.2)#,callbacks=ModelCheckpoint("model_{binary_accuracy}.hdf5")

#validation_split =0.2 設定80%訓練資料、20%驗證資料
#執行10次訓練週期
#每一批次訓練100筆資料
#verbose 顯示訓練過程

show_train_acc_history('accuracy', 'val_accuracy') 
show_train_loss_history('loss', 'val_loss')

scores = modelRNN.evaluate(x_test,  np.array(ans),verbose=1)
scores[1]
#使用test測試資料及評估準確率




from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

modelLSTM = Sequential() #建立模型

modelLSTM.add(Embedding(output_dim=32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
     input_dim=3800,  #輸入的維度是3800，也就是我們之前建立的字典是3800字
     input_length=380)) #數字list截長補短後都是380個數字



modelLSTM .add(Dropout(0.7)) #隨機在神經網路中放棄70%的神經元，避免overfitting

#建立LSTM層
modelLSTM .add(LSTM(32)) 
#建立32個神經元的LSTM層

#建立隱藏層
modelLSTM .add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
modelLSTM .add(Dropout(0.7))

#建立輸出層
modelLSTM .add(Dense(units=1,activation='sigmoid'))
 #建立一個神經元的輸出層
#查看模型摘要
modelLSTM .summary()

#訓練模型參數同RNN
modelLSTM.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

train_history = modelLSTM.fit(x_train, np.array(label), 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)


#評估模型準確率
scores = modelLSTM .evaluate(x_test,  np.array(ans),verbose=1)
scores[1]

"""

#畫出accuracy圖
show_train_acc_history('accuracy', 'val_accuracy') 
"""
"""
plt.figure()
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title("Train History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"], loc="upper left")
plt.show() 
"""
show_train_acc_history('accuracy', 'val_accuracy')
show_train_loss_history('loss', 'val_loss')


"""
epochs=range(len(train_history.history['acc']))
plt.figure()
plt.plot(epochs,train_history.history['acc'],'b',label='Training acc')
plt.plot(epochs,train_history.history['val_acc'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
#plt.savefig('/root/notebook/help/figure/model_V3.1_acc.jpg')

plt.figure()
plt.plot(epochs,train_history.history['loss'],'b',label='Training loss')
plt.plot(epochs,train_history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('/root/notebook/help/figure/model_V3.1_loss.jpg')
"""



