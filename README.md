# robotics_match_prediction

uses data from vexdb to train a neural net to predict outcome of a vrc match between two teams.

Current approach combines the output of several different feed-forward neural networks to produce a likely hood of Red team or Blue team winning.

Takes in the whole suite of statistics from VEXDB including CCWM, OPR, DPR, etc. to calculate the final output.

Currently the program is setup as a Flask server that takes HTML get requests. The program is setup in this way to be easily integratable as an API with the VScout robotics scouting application.

These requests are in the form:

```predictWithTeams/<string:teams>``` or ```predictWithStats/<string:stats>```

It is not recommended to use predictWithStats as input it quite complex. However, if you do wish to comb through it, you can look through the source code yourself.

predictWithTeams simply takes in JSON info formatted in a list. The first first two entries in a list are one team and the last two entries correspond to another team.

So for example, if you wanted to see the result of (2381C, 2381W) vs (2381Y, 2381X), you would call:
```predictWithTeams/[2381C, 2381W, 2381Y, 2381X]```

The current model is able to correctly predict the outcome of a match about 80% of the time. Newer models are in the works with hopefully even higher effectiveness and capabilites. 

To run this program, just startup server.py.

