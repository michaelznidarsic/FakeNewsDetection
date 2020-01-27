# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:27:38 2019

@author: mznid
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.preprocessing import OneHotEncoder
import math


import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn import tree




from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer



import tensorflow as tf                        # tensorflow installation and versions are finicky, prepare to use pip instead of conda, use --user, and redo with ignore installs if initially a failed installation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D, BatchNormalization






features = pd.read_csv('C:\\Users\\mznid\\Python Experiments\\featurecreation.csv')

inputs = features.iloc[:,[1,7,13,25]]

binarylist = []
for each in list(inputs.iloc[:,3]):
    if each == 'f':
        binarylist.append(0)
    else:
        binarylist.append(1)

newinputs = inputs.iloc[:,[0,1,2]]

newinputs.insert(3, "sentimentPrediction", binarylist)


sb.heatmap(newinputs.corr(), annot = True, cmap = "Blues", fmt='g') 

labels = features.iloc[:,[21,22]]

labelcolumn = list(labels.iloc[:,1])






PATERSONBINARYCORRECT = []
for each in range(0,len(labelcolumn)):
    if labelcolumn[each] == list(newinputs.iloc[:,1])[each]:
        PATERSONBINARYCORRECT.append(1)
    else:
        PATERSONBINARYCORRECT.append(0)
sum(PATERSONBINARYCORRECT) / len(PATERSONBINARYCORRECT)



dfcolumns = newinputs.columns.values



nom0 = []
nom1 = []
nom2 = []
for index, row in inputs.iterrows():
    nom0.append(str(row[0]))
    nom1.append(str(row[1]))
    nom2.append(str(row[2]))

data = {'veracityPrediction': nom0, 'binaryPrediction': nom1, 'authortypePrediction': nom2, 'sentimentPrediction': list(inputs.iloc[:,3])}
nominalinputs = pd.DataFrame(data)

enc = OneHotEncoder(handle_unknown='ignore')
X = nominalinputs
enc.fit(X)

enc.categories_

onehotnominalinputs = enc.transform(nominalinputs).toarray()

onehotnominalinputs = pd.DataFrame(onehotnominalinputs)






#### MIX MAX SCALER

scaler = MinMaxScaler()
scaler.fit(newinputs.to_numpy())
newinputs = scaler.transform(newinputs.to_numpy())
newinputs = pd.DataFrame(newinputs, columns = list(dfcolumns))  # , 'unique'   'average',   ,'extremities'



# newinputs = onehotnominalinputs


sampler = random.sample(range(0,len(labelcolumn)), round(0.75 * len(labelcolumn)))
notsampler = []
for each in range(0,len(labelcolumn)):
    if not each in sampler:
        notsampler.append(each)

train = newinputs.iloc[sampler,:]
test = newinputs.iloc[notsampler,:]

labeltrain = []
for each in sampler:
    labeltrain.append(labelcolumn[each])
labeltest = []
for each in notsampler:
    labeltest.append(labelcolumn[each])




columnlist = train.columns.values
len(columnlist)
columnselection = [0,1,2,3]#      [0,1,2,3][0,1,2,3,4,5,6,7,8,9,10]# 



RFmodel = RandomForestClassifier(n_estimators = 1000, max_depth = None, random_state = 0)
RFmodel.fit(train.iloc[:,columnselection], labeltrain)
RFprediction = RFmodel.predict(test.iloc[:,columnselection])
RFcorrect = []
for each in range(0,len(RFprediction)):
    if RFprediction[each] == labeltest[each]:
        RFcorrect.append(1)
    else:
        RFcorrect.append(0)


RFaccuracy = sum(RFcorrect)/len(RFcorrect)
print(RFaccuracy)


actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(RFprediction, dtype= "category")
confusionmatrix = pd.crosstab(actual, pred)
sb.heatmap(confusionmatrix, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Actual')







XGBmodel = xgb.XGBClassifier(learning_rate=0.5, n_estimators=140, max_depth=10,
                        min_child_weight=5, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27, booster = 'gbtree')            # gbtree, gblinear, dart
XGBmodel.fit(train.iloc[:,columnselection],labeltrain)

XGBprediction = XGBmodel.predict(test.iloc[:,columnselection])

XGBcorrect = []
for each in range(0,len(XGBprediction)):
    if XGBprediction[each] == labeltest[each]:
        XGBcorrect.append(1)
    else:
        XGBcorrect.append(0)

print(sum(XGBcorrect) / len(XGBcorrect))





actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(XGBprediction, dtype= "category")
confusionmatrix = pd.crosstab(actual, pred)
sb.heatmap(confusionmatrix, annot = True, cmap = "Blues", fmt='g')    # .iloc[[1,0,2,3],[1,0,2,3]]
plt.xlabel('Prediction')
plt.ylabel('Actual')

xgb.plot_importance(XGBmodel, importance_type = 'gain')



XGBmodel.feature_importances_

xgb.plot_tree(XGBmodel, num_trees=2)
fig = plt.gcf()
fig.set_size_inches(30, 20)
fig.savefig('tree.png')


XGBTRAINprediction = XGBmodel.predict(train)

XGBTRAINcorrect = []
for each in range(0,len(XGBTRAINprediction)):
    if XGBTRAINprediction[each] == labeltrain[each]:
        XGBTRAINcorrect.append(1)
    else:
        XGBTRAINcorrect.append(0)

print(sum(XGBTRAINcorrect) / len(XGBTRAINcorrect))









clf = DecisionTreeClassifier(min_samples_split = 3, min_samples_leaf = 2, max_depth = 8, criterion = 'entropy')

clf = clf.fit(train,labeltrain)

testpred = clf.predict(test)


print("Accuracy:", metrics.accuracy_score(labeltest, list(testpred)))
treecorrect = list(testpred == labeltest)
sum(treecorrect)/len(treecorrect)

actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(testpred, dtype= "category")
confusionmatrix = pd.crosstab(actual, pred)
sb.heatmap(confusionmatrix, annot = True, cmap = "Blues", fmt='g')    # .iloc[[1,0,2,3],[1,0,2,3]]
plt.xlabel('Prediction')
plt.ylabel('Actual')





dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names = columnlist)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('CICEROTREE2.png')
Image(graph.create_png())















NNsampletrain = train.iloc[0:10000,:]
NNsamplelabeltrain = labeltrain[0:10000]
temp = []
for each in NNsamplelabeltrain:
    if each == 'f':
        temp.append(0)
    if each == 't':
        temp.append(1)
NNsamplelabeltrain = temp        


NNsampletest = test.iloc[0:500,:]
NNsamplelabeltest = labeltest[0:500]
temp = []
for each in NNsamplelabeltest:
    if each == 'f':
        temp.append(0)
    if each == 't':
        temp.append(1)
NNsamplelabeltest = temp        




BATCH_SIZE = 3000

layer0 = tf.keras.layers.Dense(24, activation=tf.nn.relu, input_dim = 4) # hidden layer
layer1 = tf.keras.layers.Dense(12,activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(2,  activation=tf.nn.softmax) # output layer with softmax for probability/classification

model = tf.keras.Sequential([layer0, layer1, layer2])  # assemble layers into a model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])                       # assign tunings/methods to model object

history = model.fit(train.to_numpy(), np.array(labeltrain), epochs=1, verbose = True, steps_per_epoch = math.ceil(len(labeltrain)/BATCH_SIZE))  # train model object (does this alter the object itself? do we even need to assign output to a new object?)

























textinputs = pd.read_csv('C:\\Users\\mznid\\Python Experiments\\FNe_textDF.csv', header = None)

textlist = list(textinputs.iloc[:,0])

textlist[0]


os.chdir('D:\\FAKENEWSOUTPUT')


RELIABLE = pd.read_csv('RELIABLE-true.csv')

reliableindex = []
counter = 0
for each in textlist:
    if each in list(RELIABLE.iloc[:,2]):
        reliableindex.append(counter)
    counter += 1


del RELIABLE




POLITICAL = pd.read_csv('POLITICAL-mostlytrue.csv')

politicalindex = []
counter = 0
for each in textlist:
    if each in list(POLITICAL.iloc[:,2]):
        politicalindex.append(counter)
    counter += 1

del POLITICAL




BIAS = pd.read_csv('BIAS-mostlyfalse.csv')

biasindex = []
counter = 0
for each in textlist:
    if each in list(BIAS.iloc[:,2]):
        biasindex.append(counter)
    counter += 1

del BIAS



FAKE = pd.read_csv('FAKE-false.csv')

fakeindex = []
counter = 0
for each in textlist:
    if each in list(FAKE.iloc[:,2]):
        fakeindex.append(counter)
    counter += 1


del FAKE
