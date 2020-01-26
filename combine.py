import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
import numpy as np
from flask import Flask
import requests

"""
#importing the data
json_data = json.loads(open(r"data.txt", "r").read())
allData = pd.DataFrame(json_data)

#seperating it into
dataOut = allData[["redscore", "bluescore", "redwin", "bluewin"]].copy()
dataIn = allData.drop(["redscore", "bluescore", "redwin", "bluewin"], axis = 1)

dataOut = dataOut.drop(["redscore", "bluescore"], axis = 1)
#dataIn = dataIn[["r1rank", "r1opr", "r1dpr", "r2rank", "r2opr", "r2dpr", "b1rank", "b1opr", "b1dpr", "b2rank", "b2opr", "b2dpr"]].copy()
"""

#app = Flask(__name__)

#@app.route('/')
def hello():
    return "hello"

def predict(dframe):
    #take in one value at a time
    valIn = dframe.iloc[[0]]

    #0.71
    #winloss_dataIn = dataIn[["r1wins", "r1losses", "r2wins", "r2losses", "b1wins", "b1losses", "b2wins", "b2losses"]].copy()
    winloss_model = keras.models.load_model("./models/nonewinloss.h5")
    winloss_valIn = valIn[["r1wins", "r1losses", "r2wins", "r2losses", "b1wins", "b1losses", "b2wins", "b2losses"]].copy()
    winloss_prediction = winloss_model.predict(winloss_valIn.values)

    #0.71
    #rankwpapspccwm_dataIn = dataIn[["r1rank", "r1wp", "r1ap", "r1sp", "r1ccwm", "r2rank", "r2wp", "r2ap", "r2sp", "r2ccwm", "b1rank", "b1wp", "b1ap", "b1sp", "b1ccwm", "b2rank", "b2wp", "b2ap", "b2sp", "b2ccwm"]].copy()
    rankwpapspccwm_model = keras.models.load_model("./models/nonerankwpapspccwm.h5")
    rankwpapspccwm_valIn = valIn[["r1rank", "r1wp", "r1ap", "r1sp", "r1ccwm", "r2rank", "r2wp", "r2ap", "r2sp", "r2ccwm", "b1rank", "b1wp", "b1ap", "b1sp", "b1ccwm", "b2rank", "b2wp", "b2ap", "b2sp", "b2ccwm"]].copy()
    rankwpapspccwm_prediction = rankwpapspccwm_model.predict(rankwpapspccwm_valIn.values)

    #0.7
    #rankoprdpr_dataIn = dataIn[["r1rank", "r1opr", "r1dpr", "r2rank", "r2opr", "r2dpr", "b1rank", "b1opr", "b1dpr", "b2rank", "b2opr", "b2dpr"]].copy()
    rankoprdpr_model = keras.models.load_model("./models/nonerankoprdpr.h5")
    rankoprdpr_valIn = valIn[["r1rank", "r1opr", "r1dpr", "r2rank", "r2opr", "r2dpr", "b1rank", "b1opr", "b1dpr", "b2rank", "b2opr", "b2dpr"]].copy()
    rankoprdpr_prediction = rankoprdpr_model.predict(rankoprdpr_valIn.values)


    combine_arr = np.concatenate((winloss_prediction, rankwpapspccwm_prediction, rankoprdpr_prediction))
    mean_out = combine_arr.mean(axis = 0)

    return mean_out

"""
#print(mean_out)

#print("vals: ")
#print(dataIn.values[2:3])
#print("prediction: ")
#print(prediction)
#print("actual: ")
#print(dataOut.iloc[[2]])

#test_loss, test_acc = rankoprdpr_model.evaluate(dataIn.values, dataOut.values)
"""

def avg_stats(df):
    data = {"rank":df["rank"].mean(), "wins":df["wins"].mean(), "losses":df["losses"].mean(), "ties":df["ties"].mean(), "wp":df["wp"].mean(), "ap":df["ap"].mean(), "sp":df["sp"].mean(), "trsp":df["trsp"].mean(), "maxscore":df["max_score"].mean(), "opr":df["opr"].mean(), "dpr":df["dpr"].mean(), "ccwm":df["ccwm"].mean()}
    
    return pd.DataFrame(data, index = [0])

#@app.route('/predictwithstats/<string:jsonin>')
def stat_predict(stats):
    json_data = json.loads(jsonin)
    dataIn = pd.DataFrame(json_data)

    return predict(dataIn)    

#@app.route('/predictwithteams/<string:teams>')
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
    """print(team_predict("[\"15713A\", \"74947F\", \"4862B\", \"74947E\"]"))
    print(team_predict("[\"4862B\", \"4862C\", \"2381D\", \"60759A\"]"))
    print(team_predict("[\"4862D\", \"74947A\", \"4862B\", \"8716A\"]"))
    print(team_predict("[\"40999B\", \"4862B\", \"89043A\", \"74947G\"]"))
    print(team_predict("[\"2381C\", \"2381Y\", \"4862B\", \"4862A\"]"))
    print(team_predict("[\"4862B\", \"30367J\", \"4862G\", \"4862E\"]"))
    """
    link = "https://api.vexdb.io/v1/get_matches"  
    p = {"sku":"RE-VRC-19-8169"} 
    
    r = requests.get(link, p)

    data = r.json()
    
    teams_dict = {} 

    for x in range(data["size"]):
      match_info = data["result"][x]
      
      red1 = match_info["red1"]
      red2 = match_info["red2"]
      blue1 = match_info["blue1"]
      blue2 = match_info["blue2"]

      if red1 not in teams_dict:
         teams_dict[red1] = [0, 0]

      if red2 not in teams_dict:
         teams_dict[red2] = [0, 0]

      if blue1 not in teams_dict:
         teams_dict[blue1] = [0, 0]

      if blue2 not in teams_dict:
         teams_dict[blue2] = [0, 0]
      
      predict_string = "[\"" + str(red1) + "\", \"" + str(red2) + "\", \"" + str(blue1) + "\", \"" + str(blue2) + "\"]"      

      result = team_predict(predict_string)
      
      if result[0] > result[1]:
         teams_dict[red1][0] += 1
         teams_dict[red2][0] += 1
         teams_dict[blue1][1] += 1
         teams_dict[blue2][1] += 1

      if result[1] > result[0]:
         teams_dict[red1][1] += 1
         teams_dict[red2][1] += 1
         teams_dict[blue1][0] += 1
         teams_dict[blue2][0] += 1
      
      print("matchnum: " + str(x))
      print(teams_dict)

    team_list = []

    for x in teams_dict.keys():
      team_list.append((x , teams_dict[x]))       

    

     
    print("done")
    print(teams_dict)
 
    #app.run(debug=True)
