import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
import numpy as np
from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def hello():
    return "hello"

def predict(dframe):
    #take in one value at a time
    valIn = dataIn.iloc[[0]]

    winloss_model = keras.models.load_model("./models/nonewinloss.h5")
    winloss_valIn = valIn[["r1wins", "r1losses", "r2wins", "r2losses", "b1wins", "b1losses", "b2wins", "b2losses"]].copy()
    winloss_prediction = winloss_model.predict(winloss_valIn.values)

    rankwpapspccwm_model = keras.models.load_model("./models/nonerankwpapspccwm.h5")
    rankwpapspccwm_valIn = valIn[["r1rank", "r1wp", "r1ap", "r1sp", "r1ccwm", "r2rank", "r2wp", "r2ap", "r2sp", "r2ccwm", "b1rank", "b1wp", "b1ap", "b1sp", "b1ccwm", "b2rank", "b2wp", "b2ap", "b2sp", "b2ccwm"]].copy()
    rankwpapspccwm_prediction = rankwpapspccwm_model.predict(rankwpapspccwm_valIn.values)

    rankoprdpr_model = keras.models.load_model("./models/nonerankoprdpr.h5")
    rankoprdpr_valIn = valIn[["r1rank", "r1opr", "r1dpr", "r2rank", "r2opr", "r2dpr", "b1rank", "b1opr", "b1dpr", "b2rank", "b2opr", "b2dpr"]].copy()
    rankoprdpr_prediction = rankoprdpr_model.predict(rankoprdpr_valIn.values)
    combine_arr = np.concatenate((winloss_prediction, rankwpapspccwm_prediction, rankoprdpr_prediction))
    mean_out = combine_arr.mean(axis = 0)

    return mean_out

def avg_stats(df):
    data = {"rank":df["rank"].mean(), "wins":df["wins"].mean(), "losses":df["losses"].mean(), "ties":df["ties"].mean(), "wp":df["wp"].mean(), "ap":df["ap"].mean(), "sp":df["sp"].mean(), "trsp":df["trsp"].mean(), "maxscore":df["max_score"].mean(), "opr":df["opr"].mean(), "dpr":df["dpr"].mean(), "ccwm":df["ccwm"].mean()}
    
    return pd.DataFrame(data, index = [0])

@app.route('/predictwithstats/<string:jsonin>')
def stat_predict(stats):
    json_data = json.loads(jsonin)
    dataIn = pd.DataFrame(json_data)

    return predict(dataIn)    

@app.route('/predictwithteams/<string:teams>')
def team_predict(teams):
    teams_data = json.loads(teams)
    
    URL = "https://api.vexdb.io/v1/get_rankings"
    param = {"team" : teams_data[0]}

    r1 = avg_stats(pd.DataFrame(json.loads(requests.get(URL, param).text)['result'])).rename(lambda x: "r1"+x, axis="columns")
    r2 = avg_stats(pd.DataFrame(json.loads(requests.get(URL, param).text)['result'])).rename(lambda x: "r2"+x, axis="columns")
    b1 = avg_stats(pd.DataFrame(json.loads(requests.get(URL, param).text)['result'])).rename(lambda x: "b1"+x, axis="columns")
    b2 = avg_stats(pd.DataFrame(json.loads(requests.get(URL, param).text)['result'])).rename(lambda x: "b2"+x, axis="columns")

    return predict(pd.concat([r1, r2, b1, b2], axis = 1,  sort = False))
    
    
if __name__ == '__main__':
    app.run(debug=True)
