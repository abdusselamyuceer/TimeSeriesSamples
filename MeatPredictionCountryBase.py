#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:07:39 2021

@author: selam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:59:13 2021

@author: selam
"""
import pandas as pd 
import matplotlib .pyplot as plt
import numpy as np 

from numpy import hstack
from numpy import array


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("meat.csv")

print(df.info())

def QueryInDataFrame(pdf, pdctParams):
    return_df = pd.DataFrame()
    return_df = pdf
    for key, value in pdctParams.items():
        return_df = return_df[return_df[key] == value]
    return return_df    

def SplitSequenceMultiStep(sequence, nStepsIn, nStepsOut):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + nStepsIn
        out_end_ix = end_ix + nStepsOut -1
       
        if out_end_ix > len(sequence):
            break
       
        seqX, seqY = sequence[i:end_ix, :-1], sequence[end_ix-1:out_end_ix, -1]
        X.append(seqX)
        y.append(seqY)
    return array(X), array(y)
    
#print(df["LOCATION"].unique())        
#df = df[(df["LOCATION"] == "TUR" ) & (df["SUBJECT"] == "BEEF") & (df["MEASURE"] == "KG_CAP")]
#df = df.query(df["LOCATION"] == "TUR" & df["SUBJECT"] == "BEEF")
    
enc = LabelEncoder()


dct_params_beef = {"SUBJECT":"BEEF","MEASURE": "KG_CAP","LOCATION":"TUR"}
df_beef = QueryInDataFrame(df, dct_params_beef)

scaler = StandardScaler()
df_beef = df_beef[df_beef["TIME"]<2025]
df_beef["Value"] = np.transpose(scaler.fit_transform([df_beef["Value"]]))

df_beef["LOCATION"]= enc.fit_transform(df_beef["LOCATION"])

df_test =df_beef[:]
#dct_params_beef = {"LOCATION":5}
#df_test = QueryInDataFrame(df_beef[:] ,dct_params_beef)
#df_test =  df_test[(df_test["TIME"] >2019) & (df_test["TIME"] <2025)]


df_beef = df_beef[df_beef["TIME"]<2019]
df_beef.sort_values(["LOCATION","TIME"], inplace = True)



lstId = range(0,len(df_beef))

df_beef["Id"] = lstId



n_data = len(df_beef)-5

raw_seq = df_beef[["LOCATION","TIME","Value" ]][:n_data]
n_step = 8
feature_num = 2

multilen =5
X_1 = raw_seq["LOCATION"].values.reshape(len(raw_seq["LOCATION"]), 1)
X_2 = raw_seq["TIME"].values.reshape(len(raw_seq["TIME"]), 1)
y_1 = raw_seq["Value"].values.reshape(len(raw_seq["Value"]), 1)

 

dataset = hstack((X_1, X_2, y_1))

 

X, y = SplitSequenceMultiStep(dataset, n_step, multilen)
print(X.shape, y.shape)
X = X.reshape(X.shape[0], X.shape[1], feature_num)
y = y.reshape(y.shape[0], y.shape[1], 1)
model = Sequential()

model.add(LSTM(16, return_sequences=True, input_shape=(n_step, feature_num)))
model.add(Dropout(0.2)) 
model.add(LSTM(8))
model.add(Dropout(0.2)) 
model.add(Dense(multilen))

model.compile(optimizer='adam', loss='mae')

model.summary()

model.fit(X, y, epochs=30 ,batch_size=128)

x_input = array(df_test[["LOCATION","TIME"]][df_test["TIME"] >2019])

x_input = x_input.reshape((1, multilen, feature_num))

yhat = model.predict(x_input, verbose=0)


plt.plot(range(len(df_test[df_test["TIME"] <2025])),scaler.inverse_transform(df_test["Value"]), label = "gercek", color = "b")

# plt.plot(df["Hararet"][:100])


yhat = vstack((array(df_test[["Value"]][df_test["TIME"]<2020]),np.transpose(yhat)))
yhat =scaler.inverse_transform(np.transpose(yhat))
plt.plot(range(len(df_test[df_test["TIME"] <2025])), np.transpose(yhat), label = "tahmin",color="r")

plt.legend()

plt.show()  
