import tensorflow as tf
from tensorflow import keras
import pandas
import json

JSONFile = open(r"data.txt","r")
JSONText = JSONFile.read()

#print(JSONText)

jsonData = json.loads(JSONText)

matchteam = jsonData['matchteam']
print(matchteam)

for x in range(len(matchteam)):
    print(matchteam[x])


