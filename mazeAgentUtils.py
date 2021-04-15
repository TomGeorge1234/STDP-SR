import numpy as np 
import matplotlib.pyplot as plt 
import random 
import time 
from tqdm.notebook import tqdm
from matplotlib import rcParams
from datetime import datetime 
import os
from cycler import cycler
plt.style.use("seaborn")
rcParams['figure.dpi']= 300
rcParams['axes.labelsize']=5
rcParams['axes.labelpad']=2
rcParams['axes.titlepad']=3
rcParams['axes.titlesize']=5
rcParams['axes.xmargin']=0
rcParams['axes.ymargin']=0
rcParams['xtick.labelsize']=4
rcParams['ytick.labelsize']=4
rcParams['grid.linewidth']=0.5
rcParams['legend.fontsize']=4
rcParams['lines.linewidth']=0.5
rcParams.update({'figure.autolayout': True})
rcParams['xtick.major.pad']=2
rcParams['xtick.minor.pad']=2
rcParams['ytick.major.pad']=2
rcParams['ytick.minor.pad']=2
rcParams['xtick.color']='grey'
rcParams['ytick.color']='grey'
rcParams['figure.titlesize']='medium'
rcParams['axes.prop_cycle']=cycler('color', ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])


class MazeAgent():
    def __init__(self,dt=0.01,numRooms=2, roomSize=1, policy='randomWalk', state='onehot', doorsClosed=False, dx=0.1, v=1):

        #arguments
        self.numRooms = numRooms
        self.roomSize = roomSize
        self.dt = dt
        self.dx = dx
        self.v = v
        self.policy = policy
        self.state = state

        #SR learning params
        self.gamma = 0.95
        self.alpha = 0.2
        #initialise state
        self.pos = np.array([0.2,0.2])
        self.prevPos = np.array([0.2,0.2])
        self.theta = np.pi/4
        self.vel = self.v*np.array([np.sqrt(1/2),np.sqrt(1/2)])
        self.doorsClosed = doorsClosed

        #some attributes
        self.posHist = np.array([self.pos])
        self.walls = {}
        self.makeWalls()
        self.initStateVector()

    def makeWalls(self):
        assert self.numRooms in [1,2,4]
        rs = self.roomSize
        if self.numRooms == 1: 
            self.walls['room1'] = np.array([
                                    [[0,0],[0,rs]],
                                    [[0,rs],[rs,rs]],
                                    [[rs,rs],[rs,0]],
                                    [[rs,0],[0,0]]])
        elif self.numRooms == 2:
            self.walls['room1'] = np.array([
                                    [[0,0],[0,rs]],
                                    [[0,rs],[rs,rs]],
                                    [[rs,rs],[rs,0.6*rs]],
                                    [[rs,0.4*rs],[rs,0]],
                                    [[rs,0],[0,0]]])
            self.walls['room2'] = np.array([
                                    [[rs,0],[rs,0.4*rs]],
                                    [[rs,0.6*rs],[rs,rs]],
                                    [[rs,rs],[2*rs,rs]],
                                    [[2*rs,rs],[2*rs,0]],
                                    [[2*rs,0],[rs,0]]])
            self.walls['door12'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]]])
        elif self.numRooms == 4: 
            self.walls['room1'] = np.array([
                                    [[0,0],[0,rs]],
                                    [[0,rs],[0.4*rs,rs]],
                                    [[0.6*rs,rs],[rs,rs]],
                                    [[rs,rs],[rs,0.6*rs]],
                                    [[rs,0.4*rs],[rs,0]],
                                    [[rs,0],[0,0]]])
            self.walls['room2'] = np.array([
                                    [[rs,0],[rs,0.4*rs]],
                                    [[rs,0.6*rs],[rs,rs]],
                                    [[rs,rs],[1.4*rs,rs]],
                                    [[1.6*rs,rs],[2*rs,rs]],
                                    [[2*rs,rs],[2*rs,0]],
                                    [[2*rs,0],[rs,0]]])
            self.walls['room3'] = np.array([
                                    [[0,rs],[0.4*rs,rs]],
                                    [[0.6*rs,rs],[rs,rs]],
                                    [[rs,rs],[rs,1.4*rs]],
                                    [[rs,1.6*rs],[rs,2*rs]],
                                    [[rs,2*rs],[0,2*rs]],
                                    [[0,2*rs],[0,rs]]])
            self.walls['room4'] = np.array([
                                    [[rs,rs],[1.4*rs,rs]],
                                    [[1.6*rs,rs],[2*rs,rs]],
                                    [[2*rs,rs],[2*rs,2*rs]],
                                    [[2*rs,2*rs],[rs,2*rs]],
                                    [[rs,2*rs],[rs,1.6*rs]],
                                    [[rs,1.4*rs],[rs,rs]]])
            self.walls['door12'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]]])
            self.walls['door13'] = np.array([[[0.4*rs,rs],[0.6*rs,rs]]])
            self.walls['door34'] = np.array([[[rs,1.4*rs],[rs,1.6*rs]]])
            self.walls['door24'] = np.array([[[1.4*rs,rs],[1.6*rs,rs]]])

    
    def movementPolicyUpdate(self):
        self.posHist = np.vstack((self.posHist,self.pos))
        posNew = self.pos + self.vel * self.dt
        step = np.array([self.pos,posNew])
        lambda_ = self.checkWallIntercepts(step)
        self.prevPos = self.pos
        self.pos = self.pos + lambda_ * (posNew - self.pos)

        if self.policy == 'randomWalk':
            self.theta += 2*(random.random() - 0.5)*np.pi/8
            self.theta = np.mod(self.theta,2*np.pi)
            self.vel = self.v*np.array([np.cos(self.theta),np.sin(self.theta)])

    def plotHistory(self,color=None):
        fig, ax = plt.subplots(figsize=(2,2))
        ax.scatter(self.posHist[:,0],self.posHist[:,1],s=0.1,alpha=0.5,c=color)
        maxWall, minWall = [0,0],[0,0] 
        for wallObject in self.walls.keys():
            if (wallObject[:4] == 'door' and self.doorsClosed == False):
                continue

            for wall in self.walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=2)
                maxWall[0] = max(maxWall[0],max(wall[:,0]))
                minWall[0] = min(minWall[0],min(wall[:,0]))
                maxWall[1] = max(maxWall[1],max(wall[:,1]))
                minWall[1] = min(minWall[1],min(wall[:,1]))
            ax.set_ylim(bottom=minWall[1]-0.05,top=maxWall[1]+0.05)
            ax.set_xlim(left=minWall[0]-0.05,right=maxWall[0]+0.05)
        ax.set_aspect('equal')
        return fig, ax

        
    def plotPlaceField(self,number=None):
        if number == None: number = random.randint(a=0,b=len(self.stateVec_asVector)-1)
        placeField = self.M[:,number].reshape(self.stateVec_asMatrix.shape)
        fig, ax = plt.subplots(figsize=(2,2))

        ax.imshow(placeField,extent=(0,1,0,1),cmap='viridis')
        maxWall, minWall = [0,0],[0,0] 
        for wallObject in self.walls.keys():
            if (wallObject[:4] == 'door' and self.doorsClosed == False):
                continue
            for wall in self.walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=2)
                maxWall[0] = max(maxWall[0],max(wall[:,0]))
                minWall[0] = min(minWall[0],min(wall[:,0]))
                maxWall[1] = max(maxWall[1],max(wall[:,1]))
                minWall[1] = min(minWall[1],min(wall[:,1]))
            ax.set_ylim(bottom=minWall[1]-0.05,top=maxWall[1]+0.05)
            ax.set_xlim(left=minWall[0]-0.05,right=maxWall[0]+0.05)
        ax.set_aspect('equal')
        return fig, ax


    def plotGridField(self,number=None):
        if number == None: number = random.randint(a=0,b=len(self.stateVec_asVector)-1)
        eigval, eigvec = np.linalg.eig(self.M)
        eigvec = np.real(eigvec)
        gridField = eigvec[:,number].reshape(self.stateVec_asMatrix.shape)
        fig, ax = plt.subplots(figsize=(2,2))
        ax.imshow(gridField,extent=(0,1,0,1),cmap='viridis')
        maxWall, minWall = [0,0],[0,0] 
        for wallObject in self.walls.keys():
            if (wallObject[:4] == 'door' and self.doorsClosed == False):
                continue
            for wall in self.walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=2)
                maxWall[0] = max(maxWall[0],max(wall[:,0]))
                minWall[0] = min(minWall[0],min(wall[:,0]))
                maxWall[1] = max(maxWall[1],max(wall[:,1]))
                minWall[1] = min(minWall[1],min(wall[:,1]))
            ax.set_ylim(bottom=minWall[1]-0.05,top=maxWall[1]+0.05)
            ax.set_xlim(left=minWall[0]-0.05,right=maxWall[0]+0.05)
        ax.set_aspect('equal')
        return fig, ax




    def checkWallIntercepts(self,step):
        # calculates point of intercept between the line passing along the current step direction and the lines passing along the walls,
        # if this intercept lies on the current step and on the current wall this implies a "collision" 
        # this occurs iff the solution to s1 + lam_s*(s2-s1) = w1 + lam_w*(w2 - w1) satisfies 0 <= lam_s & lam_w <= 1
        s1, s2 = np.array(step[0]), np.array(step[1])
        ds = s2 - s1
        ds_perp = perp(ds)

        earliestCollision = 1 #i.e. no collision
        for wallObject in self.walls.keys():
            if (wallObject[:4] == 'door' and self.doorsClosed == False):
                continue
            for wall in self.walls[wallObject]:
                w1, w2 = np.array(wall[0]), np.array(wall[1])
                dw = w2 - w1
                dw_perp = perp(dw)
                lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / np.dot(ds, dw_perp)
                lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / np.dot(dw, ds_perp)
                if ((lam_s <= 1) and (lam_s >= 0)) and ((lam_w <= 1) and (lam_w >= 0)):
                    if lam_s < earliestCollision: 
                        earliestCollision = lam_s
                        self.wallBounce(wall)
        return earliestCollision - 0.01

        
    def wallBounce(self,wall):
        wallPerp = perp(wall[1] - wall[0])
        if np.dot(wallPerp,self.vel) <= 0:
            wallPerp = -wallPerp
        wallTheta = theta(wallPerp)
        self.theta = np.mod(wallTheta + np.pi - (self.theta - wallTheta), 2*np.pi)

    def initStateVector(self):
        if self.state == 'onehot':
            if self.numRooms == 1: 
                self.xArray = np.arange(0,self.roomSize,self.dx)
                self.yArray = np.arange(0,self.roomSize,self.dx)[::-1]
            elif self.numRooms == 2: 
                self.xArray = np.arange(0,2*self.roomSize,self.dx)
                self.yArray = np.arange(0,self.roomSize,self.dx)[::-1]
            elif self.numRooms == 4: 
                self.xArray = np.arange(0,2*self.roomSize,self.dx)
                self.yArray = np.arange(0,2*self.roomSize,self.dx)[::-1]
            self.stateVec_asMatrix = np.zeros(shape=(len(self.yArray),len(self.xArray)))
            self.stateVec_asVector = self.stateVec_asMatrix.reshape((-1))
            self.M = np.eye(len(self.stateVec_asVector))

    def TDLearningStep(self):
        currentState = self.posToState(self.pos)
        prevState = self.posToState(self.prevPos)
        delta = prevState + self.gamma*np.matmul(self.M,currentState) - np.matmul(self.M,prevState)
        self.M = self.M + self.alpha * np.outer(delta, prevState)


    def posToState(self,pos):
        xid = np.argmin(np.abs(self.xArray - pos[0]))
        yid = np.argmin(np.abs(self.yArray - pos[1]))
        state = np.zeros_like(self.stateVec_asMatrix)
        state[yid,xid] = 1
        state = state.reshape(self.stateVec_asVector.shape)
        return state




def perp(a=None):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0] 
    return b

def theta(segment):
    eps = 1e-6
    if segment.shape == (2,): 
        return np.mod(np.arctan(segment[1]/(segment[0] + eps)),2*np.pi)
    elif segment.shape == (2,2):
        return np.mod(np.arctan((segment[1][1]-segment[0][1])/(segment[1][0] - segment[0][0] + eps)), 2*np.pi)

def saveFigure(fig,saveTitle=""):
	"""saves figure to file, by data (folder) and time (name) 
	Args:
		fig (matplotlib fig object): the figure to be saved
		saveTitle (str, optional): name to be saved as. Current time will be appended to this Defaults to "".
	"""	
	today =  datetime.strftime(datetime.now(),'%y%m%d')
	if not os.path.isdir(f"./figures/{today}/"):
		os.mkdir(f"./figures/{today}/")
	figdir = f"./figures/{today}/"
	now = datetime.strftime(datetime.now(),'%H%M')
	path = f"{figdir}{saveTitle}_{now}.png"
	fig.savefig(f"{figdir}{saveTitle}_{now}.pdf", dpi=400,tight_layout=True)
	return path


