import sys 
sys.path.insert(-1,"../../")
sys.path.insert(-1,"../../tomplotlib/")
from mazeAgentUtils import *
from tomplotlib import *

fileName = sys.argv[1]

agent = MazeAgent()
agent.loadFromFile(fileName,directory="../../savedObjects/")
agent.dt = 0.001
agent.runRat(trainTime=120)
agent.saveToFile(name=fileName + "_run_full",directory="../../savedObjects/")
