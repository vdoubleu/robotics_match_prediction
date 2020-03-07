from flask import Flask
from combine import *

app = Flask(__name__)

@app.route('/predictwithstats/<string:jsonin>')
def predict_with_stats(stats):
    return stat_predict(stats)

@app.route('/predictwithteams/<string:teams>')
def predict_with_teams(teams):
    return team_predict(teams)

if __name__ == '__main__':
    app.run(debug=True)
