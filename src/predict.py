import tensorflow as tf
from tensorflow import keras
import pandas as pd
import json
from sklearn.model_selection import train_test_split

JSONFile = open(r"data.txt","r")
JSONText = JSONFile.read()

jsonData = json.loads(JSONText)

dataIn = pd.DataFrame(jsonData)

print("all dataIn")
print(dataIn)

#makes new tables filled with the expected outs
dataOut = dataIn[["redscore", "bluescore", "redwin", "bluewin"]].copy()
dataIn = dataIn.drop(["redscore", "bluescore", "redwin", "bluewin"], axis=1)

dataInTrain, dataInTest, dataOutTrain, dataOutTest = train_test_split(dataIn, dataOut, test_size = 0.2)

print("dataOut")
print(dataOut)

print("dataIn")
print(dataIn)

#dataIn.drop(["redscore", "bluescore"], axis=1)

#modCol()

model = keras.Sequential([
    keras.layers.Input(shape=(len(dataInTrain.index), )),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(2, activation="softmax")])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(dataInTrain, dataOutTrain, epochs = 2)

test_loss, test_acc = model.evaluate(datainTest, dataOutTest)



def modCol(df, colId, func):
    df[coldId] = df[coldId].apply(func)
    
