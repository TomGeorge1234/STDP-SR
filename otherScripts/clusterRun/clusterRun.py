import sys 
sys.path.insert(-1,"../../")
sys.path.insert(-1,"../../tomplotlib/")
from mazeAgentUtils import *
from tomplotlib import *

fileName = sys.argv[1]

agent = MazeAgent()
agent.loadFromFile(fileName,directory="../../savedObjects/")
agent.runRat(trainTime=1)
print("YEET")
# agent.saveToFile(name=fileName)