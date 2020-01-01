import tensorflow as tf
from tensorflow import keras
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def modCol(df, colId, func):
    df[colId] = df[colId].apply(func)
 
def normCol(df, colId):
    pdata = preprocessing.normalize([np.array(df[colId])])
    df.loc[:,colId] = pdata[0]

def normDf(df, idlst):
    for x in idlst:
        normCol(df, x)

def normEntireDf(df):
    for x in df.columns.values:
        normCol(df, x)

def modEntireDf(df, func):
    for x in df.columns.values:
        modCol(df, x, func)

def main():
    #load json data in dataFrame
    JSONFile = open(r"data.txt","r")
    JSONText = JSONFile.read()
    jsonData = json.loads(JSONText)
    dataIn = pd.DataFrame(jsonData)

    #makes new tables filled with the expected outs
    dataOut = dataIn[["redscore", "bluescore", "redwin", "bluewin"]].copy()
    dataIn = dataIn.drop(["redscore", "bluescore", "redwin", "bluewin"], axis=1)

    #decrease size of data using 1/x
    #modEntireDf(dataIn, lambda x: 1/x if x != 0 else 10000)
    modCol(dataOut, "redscore", lambda x: sigmoid(x))
    modCol(dataOut, "bluescore", lambda x: sigmoid(x))
    
    dataOut = dataOut.drop(["redscore", "bluescore"], axis=1)
    #dataIn = dataIn.drop(["r1maxscore", "r2maxscore", "b1maxscore", "b2maxscore"], axis=1)
    dataIn = dataIn[["r1rank", "r1opr", "r1dpr", "r2rank", "r2opr", "r2dpr", "b1rank", "b1opr", "b1dpr", "b2rank", "b2opr", "b2dpr"]].copy()
     
    dataInTrain, dataInTest, dataOutTrain, dataOutTest = train_test_split(dataIn, dataOut, test_size = 0.2)

    
    model = keras.Sequential([
        keras.layers.Input(shape=(len(dataInTrain.columns), )),
        keras.layers.Dense(64, activation="elu"),
        keras.layers.Dense(4, activation="elu"),
        keras.layers.Dense(2, activation="softmax")])

    model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

    model.fit(dataInTrain.values, dataOutTrain.values, epochs = 7)
     
    test_loss, test_acc = model.evaluate(dataInTest.values, dataOutTest.values)
   
    model.save("nonerankoprdpr.h5") 
if __name__ == "__main__":
    main()
   
