import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
import numpy as np

#importing the data
json_data = json.loads(open(r"data.txt", "r").read())
allData = pd.DataFrame(json_data)

#seperating it into
dataOut = allData[["redscore", "bluescore", "redwin", "bluewin"]].copy()
dataIn = allData.drop(["redscore", "bluescore", "redwin", "bluewin"], axis = 1)

dataOut = dataOut.drop(["redscore", "bluescore"], axis = 1)
dataIn = dataIn[["r1rank", "r1wp", "r1ap", "r1sp", "r1ccwm", "r2rank", "r2wp", "r2ap", "r2sp", "r2ccwm", "b1rank", "b1wp", "b1ap", "b1sp", "b1ccwm", "b2rank", "b2wp", "b2ap", "b2sp", "b2ccwm"]].copy()

"""
#take in one value at a time
valIn = dataIn.iloc[[2]]

#0.71
#winloss_dataIn = dataIn[["r1wins", "r1losses", "r2wins", "r2losses", "b1wins", "b1losses", "b2wins", "b2losses"]].copy()
winloss_model = keras.models.load_model("nonewinloss.h5")
winloss_valIn = valIn[["r1wins", "r1losses", "r2wins", "r2losses", "b1wins", "b1losses", "b2wins", "b2losses"]].copy()
winloss_prediction = winloss_model.predict(winloss_valIn.values)

print(winloss_prediction)

#0.71
#rankwpapspccwm_dataIn = dataIn[["r1rank", "r1wp", "r1ap", "r1sp", "r1ccwm", "r2rank", "r2wp", "r2ap", "r2sp", "r2ccwm", "b1rank", "b1wp", "b1ap", "b1sp", "b1ccwm", "b2rank", "b2wp", "b2ap", "b2sp", "b2ccwm"]].copy()
rankwpapspccwm_model = keras.models.load_model("nonerankwpapspccwm.h5")
rankwpapspccwm_valIn = valIn[["r1rank", "r1wp", "r1ap", "r1sp", "r1ccwm", "r2rank", "r2wp", "r2ap", "r2sp", "r2ccwm", "b1rank", "b1wp", "b1ap", "b1sp", "b1ccwm", "b2rank", "b2wp", "b2ap", "b2sp", "b2ccwm"]].copy()
rankwpapspccwm_prediction = rankwpapspccwm_model.predict(rankwpapspccwm_valIn.values)
"""
"""
combine_arr = np.concatenate((winloss_prediction, rankwpapspccwm_prediction))
mean_out = combine_arr.mean(axis = 0)
print(mean_out)

#print("vals: ")
#print(dataIn.values[18:19])
#print("prediction: ")
#print(prediction)
print("actual: ")
print(dataOut.iloc[[2]])
"""
test_loss, test_acc = rankwpapspccwm_model.evaluate(dataIn.values, dataOut.values)
