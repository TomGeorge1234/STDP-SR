import numpy as np 
import matplotlib.pyplot as plt 
import random 
import matplotlib
import pandas as pd 
from tqdm.notebook import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
from datetime import datetime 
import time 
import os
from cycler import cycler
from scipy.sparse import csr_matrix
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
rcParams['image.cmap'] = 'inferno'

defaultParams = { 
          'mazeType'           : 'oneRoom',
          'stateType'          : 'onehot',
          'roomSize'           : 1,
          'dt'                 : 0.01,
          'dx'                 : 0.01,     
          'tau'                : 2,
          'TDdx'               : 0.02,
          'nCells'             : None,
          'velScale'           : 0.16,
          'rotVelScale'        : None,
          'initPos'            : None,  
          'initDir'            : None,
          'sigma'              : 0.3,
          'placeCellThreshold' : 0.5, 
          'gridCellThreshold' : 0,
          'doorsClosed'        : True

}
class MazeAgent():
    def __init__(self,
                params)
        #arguments
        self.setParams(defautParams)
        self.setParmas(params)
        self.initialise()

        self.mazeType = mazeType 
        self.roomSize = roomSize #scale the maze, in metres
        self.dt = dt
        self.dx = dx
        self.movementPolicy = movementPolicy
        self.stateType = stateType
        self.tau = tau 
        self.alpha = alpha
        self.velocityScale = 0.16
        self.rotationalVelocityScale = 1*np.pi

        #initialise state. the agent has a position, a direction and a velocity at all times 
        self.vel = self.velocityScale
        self.t = 0
        self.runID = 0



        #initialise maze
        self.initialiseMaze()

        self.pcThreshold = 0
        self.avdelta = []

    def updateParams(self, params):
        self.mazeType = params.get('mazeType',self.mazeType)
        self.stateType = params.get('stateType',self.stateType)
        self.roomSize = params.get('roomSize',self.roomSize)
        self.dt = params.get('dt',self.dt)
        self.dx = params.get('dx',self.dx)
        self.tau = params.get('tau',self.tau)
        self.TDdx = params.get('TDdx',self.TDdx)
        self.nCells = params.get('nCells',self.nCells)
        self.velScale = params.get('velocityScale',self.velScale)
        self.rotVelScale = params.get('rotVelocity',self.rotVelScale)
        self.placeCellThreshold = params.get('placeCellThreshold',self.placeCellThreshold)
        self.gridCellThreshold = params.get('gridCellThreshold',self.gridCellThreshold)
        self.sigma = params.get('sigma',self.sigma)
        self.initPos = params.get('initPos',self.initPos)
        self.initDir = params.get('vel',self.vel)
        self.doorsClosed = params.get('doorsClosed',self.doorsClosed)


    def initialiseRat(self, paramDict): #should only be called once at the start 
        #set pos/vel
        self.pos = self.initPos
        self.vel = self.velScale
        self.dir = self.initDir

        #make maze 
        self.walls = getWalls(mazeType=self.mazeType, roomSize=self.roomSize)

        if self.initPos is None: 
            ex = self.extent
            self.initPos = np.array([ex[0] + 0.2*(ex[1]-ex[0]),
                                     ex[2] + 0.2*(ex[3]-ex[2])])
        if self.rotVelScale is None: 
            if self.mazeType == 'loop' or self.mazeTpye == 'longCorridor':
                self.rotVelScale = np.pi
            else: 
                self.rotVelScale = 
        #set cells

        #get discreteCoordintates and states 

        #initialise history dataframes
        self.history = pd.DataFrame(columns = ['t','pos','color','runID']) 
        self.snapshots = pd.DataFrame(columns = ['t','M','mazeState'])






    def runRat(self, trainTime=10, plotColor='C0',saveEvery=1, decayalpha=False):
        steps = int(trainTime * 60 / self.dt)
        hist_t, hist_pos, hist_color, hist_runID = [0]*steps, [0]*steps, [plotColor]*steps, [self.runID]*steps
        lastTDstep, n_TD, distanceToTD = 0, 0, np.random.exponential(0.02) #2cm scale
        hist_pos[0], hist_t[0] = self.pos, self.t

        saveSnapshots=False
        if saveEvery is not None:
            saveSnapshots, saveFreq = True, int(saveEvery * 60 / self.dt)

        for i in tqdm(range(1,steps)): #main training loop 

            if (saveSnapshots is True) and (i % saveFreq == 0):
                snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'mazeState':[self.mazeState]})
                self.snapshots = self.snapshots.append(snapshot)


            self.movementPolicyUpdate()
            self.t += self.dt
            hist_pos[i], hist_t[i] = self.pos, self.t

            alpha_ = self.alpha
            if decayalpha == True: 
                alpha_ *= np.exp(-(self.t/60)/(0.5*trainTime))

            #maybe do TD step
            if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= distanceToTD: #if it's moved over 2cm meters from last step 
                dt_TD = self.t - hist_t[lastTDstep]
                self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dt_TD, tau=self.tau, alpha=alpha_)
                lastTDstep = i
                n_TD += 1
                distanceToTD = np.random.exponential(0.02)
                hist_color[i] = 'C1'
            else:
                # print("notd")
                pass

        self.runID += 1

        runHistory = pd.DataFrame({'t':hist_t, 'pos':hist_pos, 'color':hist_color, 'runID':hist_runID})
        self.history = self.history.append(runHistory)
        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = self.snapshots.append(snapshot)

        self.gridFields = self.getGridFields(self.M)
        self.placeFields = self.getPlaceFields(self.M)

        print(n_TD)


    def TDLearningStep(self, pos, prevPos, dt, tau, alpha,  learnsparse=False):

        state = self.posToState(pos,stateType=self.stateType)
        prevState = self.posToState(prevPos,stateType=self.stateType)

        #onehot optimised TD learning 
        if (self.stateType == 'onehot'): 

            s_t = np.argwhere(prevState)[0][0]
            s_tplus1 = np.argwhere(state)[0][0]
            delta = state + (tau / dt) * ((1 - dt/tau) * self.M[:,s_tplus1] - self.M[:,s_t])
            self.M[:,s_t] = self.M[:,s_t] + alpha * delta

            if learnsparse == True:
                state_s, prevState_s = csr_matrix(state), csr_matrix(prevState)
                delta = state_s.T + (tau / dt)* (np.dot(self.M_sparse, ((1 - dt/tau) * state_s - prevState_s).T))
                self.M_sparse = self.M_sparse + alpha * np.dot(delta,state_s)
        
        #normal TD learning 
        else:
            # delta = state + (tau / dt) * (self.M @ ((1 - dt/tau)*state - prevState))
            # self.M = self.M + alpha * np.outer(delta, state)

            delta = prevState + (tau / dt) * (self.M @ (state - (1 + dt/tau)*prevState))
            self.M = self.M + alpha * np.outer(delta, prevState)
        
        self.avdelta.append(np.linalg.norm(delta))

    def movementPolicyUpdate(self):

        proposedNewPos = self.pos + self.vel * self.dir * self.dt
        proposedStep = np.array([self.pos,proposedNewPos])
        checkResult = self.checkWallIntercepts(proposedStep)

        if self.movementPolicy == 'randomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                randomTurnSpeed = np.random.normal(0,self.rotationalVelocityScale)
                self.turn(turnAngle=randomTurnSpeed*self.dt)
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.wallBounce(wall)
        
        if self.movementPolicy == 'raudies':
            if checkResult[0] == 'noImmediateCollision':
                self.pos = proposedNewPos
                self.vel = np.random.rayleigh(self.velocityScale)
                randomTurnSpeed = np.random.normal(0,self.rotationalVelocityScale)
                self.turn(turnAngle=randomTurnSpeed*self.dt)
            if checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.wallBounce(wall)
            if checkResult[0] == 'collisionAhead':
                wall = checkResult[1]
                self.wallFollow(wall)
                randomTurnSpeed = np.random.normal(0,self.rotationalVelocityScale)
                self.turn(turnAngle=randomTurnSpeed*self.dt)

        if self.movementPolicy == 'windowsScreensaver':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.wallBounce(wall)
        
        if self.mazeType == 'loop':
            self.pos[0] = self.pos[0] % self.roomSize

    def initialiseMaze(self):

        self.doorState = 'closed' 
        self.walls = {}
        self.mazeState = {}
        rs = self.roomSize
        dx = self.dx

        if self.mazeType == 'oneRoom': 
            self.walls['room1'] = np.array([
                                    [[0,0],[0,rs]],
                                    [[0,rs],[rs,rs]],
                                    [[rs,rs],[rs,0]],
                                    [[rs,0],[0,0]]])

            self.xArray = np.arange(dx/2,rs,dx)
            self.yArray = np.arange(dx/2,rs,dx)[::-1]
            self.extent = (0,rs,0,rs)

        if self.mazeType == 'twoRooms': 
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
            self.walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]]])

            self.xArray = np.arange(dx/2,2*rs,dx)
            self.yArray = np.arange(dx/2,rs,dx)[::-1]
            self.extent = (0,2*rs,0,rs)

        if self.mazeType == 'fourRooms': 
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
            self.walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]],
                                            [[0.4*rs,rs],[0.6*rs,rs]],
                                            [[rs,1.4*rs],[rs,1.6*rs]],
                                            [[1.4*rs,rs],[1.6*rs,rs]]])

            self.xArray = np.arange(dx/2,2*rs,dx)
            self.yArray = np.arange(dx/2,2*rs,dx)[::-1]
            self.extent = (0,2*rs,0,2*rs)
        
        if self.mazeType == 'twoRoomPassage':
            self.walls['room1'] = np.array([
                                    [[0,0],[rs,0]],
                                    [[rs,0],[rs,rs]],
                                    [[rs,rs],[0.75*rs,rs]],
                                    [[0.25*rs,rs],[0,rs]],
                                    [[0,rs],[0,0]]])
            self.walls['room2'] = np.array([
                                    [[rs,0],[2*rs,0]],
                                    [[2*rs,0],[2*rs,rs]],
                                    [[2*rs,rs],[1.75*rs,rs]],
                                    [[1.25*rs,rs],[rs,rs]],
                                    [[rs,rs],[rs,0]]])
            self.walls['room3'] = np.array([
                                    [[0,rs],[0,1.4*rs]],
                                    [[0,1.4*rs],[2*rs,1.4*rs]],
                                    [[2*rs,1.4*rs],[2*rs,rs]]])
            self.walls['doors'] = np.array([[[0.25*rs,rs],[0.75*rs,rs]],
                                            [[1.25*rs,rs],[1.75*rs,rs]]])


            self.xArray = np.arange(dx/2,2*rs,dx)
            self.yArray = np.arange(dx/2,1.2*rs,dx)[::-1]
            self.extent = (0,2*rs,0,1.2*rs)

        if self.mazeType == 'longCorridor':
            self.walls['room1'] = np.array([
                                    [[0,0],[0,rs]],
                                    [[0,rs],[rs,rs]],
                                    [[rs,rs],[rs,0]],
                                    [[rs,0],[0,0]]])
            self.walls['longbarrier'] = np.array([
                                    [[0.1*rs,0],[0.1*rs,0.9*rs]],
                                    [[0.2*rs,rs],[0.2*rs,0.1*rs]],
                                    [[0.3*rs,0],[0.3*rs,0.9*rs]],
                                    [[0.4*rs,rs],[0.4*rs,0.1*rs]],
                                    [[0.5*rs,0],[0.5*rs,0.9*rs]],
                                    [[0.6*rs,rs],[0.6*rs,0.1*rs]],
                                    [[0.7*rs,0],[0.7*rs,0.9*rs]],
                                    [[0.8*rs,rs],[0.8*rs,0.1*rs]],
                                    [[0.9*rs,0],[0.9*rs,0.9*rs]]
            ])
            self.pos = np.array([0.05,0.05])
            self.dir = np.array([0,1])

            self.xArray = np.arange(dx/2,rs,dx)
            self.yArray = np.arange(dx/2,rs,dx)[::-1]
            self.extent = (0,rs,0,rs)

        if self.mazeType == 'rectangleRoom': 
            ratio = np.pi/2.8
            self.walls['room1'] = np.array([
                                    [[0,0],[0,rs]],
                                    [[0,rs],[ratio*rs,rs]],
                                    [[ratio*rs,rs],[ratio*rs,0]],
                                    [[ratio*rs,0],[0,0]]])

            self.xArray = np.arange(dx/2,ratio*rs,dx)
            self.yArray = np.arange(dx/2,rs,dx)[::-1]
            self.extent = (0,ratio*rs,0,rs)
        
        if self.mazeType == 'loop': 
            height = 0.05
            self.walls['room'] = np.array([
                                    [[0,0],[rs,0]],
                                    [[0,height*rs],[rs,height*rs]]])
            self.walls['doors'] = np.array([
                                    [[0,0],[0,height*rs]],
                                    [[rs,0],[rs,height*rs]]])
            self.pos = np.array([0.05,0.05])
            self.dir = np.array([1,0])
            self.xArray = np.arange(dx/2,rs,dx)
            self.yArray = np.arange(dx/2,height*rs,dx)[::-1]
            self.extent = (0,rs,0,height*rs)
        
        x_mesh, y_mesh = np.meshgrid(self.xArray,self.yArray)
        coordinate_mesh = np.array([x_mesh, y_mesh])
        self.discreteCoords = np.swapaxes(np.swapaxes(coordinate_mesh,0,1),1,2) #an array of discretised position coords over entire map extent 

        self.mazeState['walls'] = self.walls
        self.mazeState['extent'] = self.extent


        if self.stateType == 'onehot': 
            self.stateSize = len(self.xArray) * len(self.yArray)
            self.stateVec_asMatrix = np.zeros(shape=self.discreteCoords.shape[:-1])
            self.stateVec_asVector = self.stateVec_asMatrix.reshape((-1))
            self.M = np.eye(self.stateSize) / self.stateSize
            # self.M = np.zeros(shape=(self.stateSize,self.stateSize))
            self.M_sparse = csr_matrix(self.M)

        if self.stateType == 'placeFields' or self.stateType == 'circles':
            self.stateSize = self.nCells
            xcentres = np.random.uniform(self.extent[0],self.extent[1],self.nCells)
            ycentres = np.random.uniform(self.extent[2],self.extent[3],self.nCells)
            self.centres = np.array([xcentres,ycentres]).T
            # self.centres[-1] = np.array([0.5*self.extent[1],0.5*self.extent[3]]) #add in a known centre
            # self.centres[-2] = np.array([0.95*self.extent[1],0.5*self.extent[3]]) #add in a known centre
            # self.centres[-3] = np.array([0.05*self.extent[1],0.5*self.extent[3]]) #add in a known centre
            # self.centres[-4] = np.array([0.95*self.extent[1],0.95*self.extent[3]]) #add in a known centre
            # self.centres[-5] = np.array([0.05*self.extent[1],0.05*self.extent[3]]) #add in a known centre
            # self.centres[-6] = np.array([0.5*self.extent[1],0.95*self.extent[3]]) #add in a known centre
            inds = self.centres[:,0].argsort()
            self.centres = self.centres[inds]
            # self.M = np.random.randn(self.stateSize,self.stateSize) / self.stateSize**2
            # self.M = np.zeros(shape=(self.stateSize,self.stateSize))
            self.M = np.eye(self.stateSize) / self.stateSize
  
        if self.stateType == 'fourier':
            self.stateSize = self.nCells
            self.kVectors = np.random.rand(self.nCells,2) - 0.5
            self.kVectors /= np.linalg.norm(self.kVectors, axis=1)[:,None]
            self.kFreq = 2*np.pi / np.random.uniform(0.01,1,size=(self.nCells))
            self.phi = np.random.uniform(0,2*np.pi,size=(self.nCells))
            self.M = np.eye(self.stateSize) / self.stateSize
        
        self.toggleDoors(doorState = self.doorState)
        # self.discreteStates = self.posToState(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 

    def turn(self, turnAngle):
        theta_ = theta(self.dir)
        theta_ += turnAngle
        theta_ = np.mod(theta_, 2*np.pi)
        self.dir = np.array([np.cos(theta_),np.sin(theta_)])

    def toggleDoors(self, doorState = None): #this function could be made more advanced to toggle more maze options
        if doorState == 'open': self.doorState = 'open'
        elif doorState == 'closed': self.doorState = 'closed'
        elif doorState == None: 
            currentDoorState == self.doorState
            if currentDoorState == 'open': self.doorState = 'closed'
            elif currentDoorState == 'closed': self.doorState = 'open'
        
        walls = self.walls.copy()
        if self.doorState == 'open': 
            del walls['doors']
            self.mazeState['walls'] = walls
        elif self.doorState == 'closed': 
            self.mazeState['walls'] = walls

        self.discreteStates = self.posToState(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 

        return self.mazeState['walls']



    def checkWallIntercepts(self,proposedStep): #proposedStep = [pos,proposedNextPos]
        s1, s2 = np.array(proposedStep[0]), np.array(proposedStep[1])
        pos = s1
        ds = s2 - s1
        stepLength = np.linalg.norm(ds)
        ds_perp = perp(ds)

        collisionList = [[],[]]
        futureCollisionList = [[],[]]

        #check if the current step results in a collision 
        walls = self.mazeState['walls']

        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                w1, w2 = np.array(wall[0]), np.array(wall[1])
                dw = w2 - w1
                dw_perp = perp(dw)

                # calculates point of intercept between the line passing along the current step direction and the lines passing along the walls,
                # if this intercept lies on the current step and on the current wall (0 < lam_s < 1, 0 < lam_w < 1) this implies a "collision" 
                # if it lies ahead of the current step and on the current wall (lam_s > 1, 0 < lam_w < 1) then we should "veer" away from this wall
                # this occurs iff the solution to s1 + lam_s*(s2-s1) = w1 + lam_w*(w2 - w1) satisfies 0 <= lam_s & lam_w <= 1
                lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / np.dot(ds, dw_perp)
                lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / np.dot(dw, ds_perp)

                #there are two situations we need to worry about: 
                # • 0 < lam_s < 1 and 0 < lam_w < 1: the collision is ON the current proposed step . Do something immediately.
                # • lam_s > 1     and 0 < lam_w < 1: the collision is on the current trajectory, some time in the future. Maybe do something. 
                if (0 <= lam_s <= 1) and (0 <= lam_w <= 1):
                    collisionList[0].append(wall)
                    collisionList[1].append([lam_s,lam_w])
                    continue

                if (lam_s > 1) and (0 <= lam_w <= 1):
                    if lam_s * stepLength <= 0.05: #if the future collision is under 3cm away
                        futureCollisionList[0].append(wall)
                        futureCollisionList[1].append([lam_s,lam_w])
                        continue
        
        if len(collisionList[0]) != 0:
            wall_id = np.argmin(np.array(collisionList[1])[:,0]) #first wall you collide with on step 
            wall = collisionList[0][wall_id]
            return ('collisionNow', wall)
        
        elif len(futureCollisionList[0]) != 0:
            wall_id = np.argmin(np.array(futureCollisionList[1])[:,0]) #first wall you would collide with along current step 
            wall = futureCollisionList[0][wall_id]
            return ('collisionAhead', wall)
        
        else:
            return ('noImmediateCollision',None)
                  
    def wallBounce(self,wall):
        wallPerp = perp(wall[1] - wall[0])
        if np.dot(wallPerp,self.dir) <= 0:
            wallPerp = -wallPerp #it is now the perpendicular with smallest angle to dir 
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,self.dir) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar, wallPerp = wallPar/np.linalg.norm(wallPar), wallPerp/np.linalg.norm(wallPerp) #normalise
        dir_ = wallPar * np.dot(self.dir,wallPar) - wallPerp * np.dot(self.dir,wallPerp)
        self.dir = dir_/np.linalg.norm(dir_)

    def wallFollow(self,wall):
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,self.dir) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar = wallPar/np.linalg.norm(wallPar)
        dir_ = wallPar * np.dot(self.dir,wallPar)
        self.dir = dir_/np.linalg.norm(dir_)
        return

    def getPlaceFields(self, M):
        placeFields = np.einsum("ij,klj->ikl",M,self.discreteStates)
        threshold_ = self.pcThreshold
        threshold = threshold_*np.amax(placeFields,axis=(1,2))[:,None,None]
        placeFields = np.maximum(0,placeFields - threshold)
        return placeFields

    def getGridFields(self, M, alignToFinal=False):
        _, eigvecs = np.linalg.eig(M) #"v[:,i] is the eigenvector corresponding to the eigenvalue w[i]"
        eigvecs = np.real(eigvecs)
        gridFields = np.einsum("ij,kli->jkl",eigvecs,self.discreteStates)
        if alignToFinal == True:
            grids_final_flat = np.reshape(self.gridFields,(self.stateSize,-1))
            grids_flat = np.reshape(gridFields,(self.stateSize,-1))
            dotprods = np.empty(grids_flat.shape[0])
            for i in range(len(dotprods)):
                dotprodsigns = np.sign(np.diag(np.matmul(grids_final_flat,grids_flat.T)))
                gridFields *= dotprodsigns[:,None,None]
        gridFields = np.maximum(0,gridFields)
        return gridFields
    
    def posToState(self, pos, stateType=None): #pos is an [n1, n2, n3, ...., 2] array of 2D positions
        """Takes an array of 2D positions of size (n1, n2, n3, ..., 2)
        retyrns the state vector for each of these positions of size (n1, n2, n3, ..., N) where N is the size of the state vector
        Args:
            pos ([type]): [description]

        Returns:
            [type]: [description]
        """ 
        if stateType == None: stateType = self.stateType

        if stateType == 'onehot':     
            x_s = pos[..., 0][...,None]
            y_s = pos[..., 1][...,None]
            x_s = np.tile(x_s, len(x_s.shape[:-1]) * (1,) + (len(self.xArray),))
            y_s = np.tile(y_s, len(y_s.shape[:-1]) * (1,) + (len(self.yArray),))
            x_s = x_s - self.xArray
            y_s = y_s - self.yArray
            x_s = np.argmin(np.abs(x_s),axis=-1)
            y_s = np.argmin(np.abs(y_s),axis=-1)
            onehotcoord = y_s * len(self.yArray) + x_s
            states = (np.arange(self.stateSize) == onehotcoord[...,None]).astype(int)
        
        if stateType == 'placeFields':
            centres = self.centres
            pos = np.expand_dims(pos,-2)
            diff = np.abs((centres - pos))
            if (self.mazeType == 'loop') and (self.doorState == 'open'):
                diff_aroundloop = diff.copy()
                diff_aroundloop[..., 0] = (self.extent[1]-self.extent[0]) - diff_aroundloop[..., 0]
                dev = np.linalg.norm(diff,axis=-1)
                dev_aroundloop = np.linalg.norm(diff_aroundloop,axis=-1)
                states = (1/self.sigma1)*np.exp(-dev**2 / (2*self.sigma1**2)) - (1/self.sigma2)*np.exp(-dev**2 / (2*self.sigma2**2))
                states_aroundloop = (1/self.sigma1)*np.exp(-dev_aroundloop**2 / (2*self.sigma1**2)) - (1/self.sigma2)*np.exp(-dev_aroundloop**2 / (2*self.sigma2**2))
                states = states + states_aroundloop
            else:
                dev = np.linalg.norm(diff,axis=-1)
                states = (1/self.sigma1)*np.exp(-dev**2 / (2*self.sigma1**2)) - (1/self.sigma2)*np.exp(-dev**2 / (2*self.sigma2**2))
            states = states/states.shape[-1]
        if stateType == 'bvc':
            pass
        
        if stateType == 'fourier':
            phase = np.matmul(pos,self.kVectors.T) * self.kFreq + self.phi
            fr = np.cos(phase)
            states = fr 

        if stateType == 'circles':
            centres = self.centres
            pos = np.expand_dims(pos,-2)
            dev = np.linalg.norm((centres - pos),axis=-1)
            states = dev
            states = np.where(states > self.sigma1, 0, 1)
            # states = states / len(states)



        return states
        
def getWalls(mazeType, roomSize, **kwargs)

class Visualiser():
    def __init__(self, mazeAgent):
        self.mazeAgent = mazeAgent
        self.snapshots = mazeAgent.snapshots
        self.history = mazeAgent.history

    def plotMazeStructure(self,fig=None,ax=None,hist_id=-1):
        snapshot = self.snapshots.iloc[hist_id]
        extent, walls = snapshot['mazeState']['extent'], snapshot['mazeState']['walls']

        if (fig, ax) == (None, None): 
            fig, ax = plt.subplots(figsize=(2*(extent[1]-extent[0]),2*(extent[3]-extent[2])))
        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=2)
            ax.set_xlim(left=extent[0]-0.05,right=extent[1]+0.05)
            ax.set_ylim(bottom=extent[2]-0.05,top=extent[3]+0.05)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        return fig, ax
    
    def plotTrajectory(self,hist_id=-1,endtime=2):
        fig, ax = self.plotMazeStructure(hist_id=hist_id)
        endid = self.history['t'].sub(endtime*60).abs().to_numpy().argmin()
        trajectory = np.stack(self.history['pos'][:endid])
        color = np.stack(self.history['color'][:endid])
        ax.scatter(trajectory[:,0],trajectory[:,1],s=0.2,alpha=0.7,c=color)
        saveFigure(fig, "trajectory")
        return fig, ax

    
    def plotM(self,hist_id=-1):
        fig, ax = plt.subplots(figsize=(2,2))
        M = self.snapshots.iloc[hist_id]['M']
        im = ax.imshow(M)
        fig.colorbar(im, ax=ax)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        saveFigure(fig, "M")
        return fig, ax


    def addTimestamp(self, fig, ax, i=-1):
        t = self.mazeAgent.saveHist[i]['t']
        ax.text(x=0, y=0, t="%.2f" %t)

    def plotPlaceField(self, hist_id=-1, time=None, fig=None, ax=None, number=None, show=True, animationCall=False, plotTimeStamp=False):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time).abs().to_numpy().argmin()
        #if a figure/ax objects are passed, clear the axis and replot the maze
        if (ax is not None) and (fig is not None): 
            ax.clear()
            self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
        # else if they are not passed plot the maze
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        
        if number == None: number = random.randint(a=0,b=self.mazeAgent.stateSize-1)
        
        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        t = snapshot['t'] / 60
        extent = snapshot['mazeState']['extent']
        placeFields = self.mazeAgent.getPlaceFields(M=M)
        ax.imshow(placeFields[number],extent=extent,interpolation=None)
        if plotTimeStamp == True: 
            ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        saveFigure(fig, "placeField")
        return fig, ax
    
    def plotReceptiveField(self, number=None, hist_id=-1, fig=None, ax=None, show=True):
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        if number == None: number = random.randint(a=0,b=self.mazeAgent.stateSize-1)
        extent = self.mazeAgent.extent
        rf = self.mazeAgent.discreteStates[..., number]
        ax.imshow(rf,extent=extent,interpolation=None)
        if show==False:
            plt.close(fig)
        saveFigure(fig, "receptiveField")
        return fig, ax
    

    def plotGridField(self, hist_id=-1, time=None, fig=None, ax=None, number=0, show=True, animationCall=False, plotTimeStamp=False):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()

        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        t = snapshot['t'] / 60
        extent = snapshot['mazeState']['extent']
        if hist_id == -1 and animationCall == False:
            gridFields = self.mazeAgent.gridFields
        else:
            gridFields = self.mazeAgent.getGridFields(M=M,alignToFinal=True)

        if number == 'many': 
            fig = plt.figure(figsize=(10, 10*((extent[3]-extent[2])/(extent[1]-extent[0]))))
            gs = matplotlib.gridspec.GridSpec(6, 6, hspace=0.1, wspace=0.1)
            c=0
            numberstoplot = np.concatenate((np.array([0,1,2,3,4,5]),np.geomspace(6,gridFields.shape[0]-1,30).astype(int)))
            for i in range(6):
                for j in range(6):
                    ax = plt.subplot(gs[i,j])
                    ax.imshow(gridFields[numberstoplot[c]],extent=extent,interpolation=None)
                    ax.grid(False)
                    ax.axis('off')
                    ax.text(extent[1]-0.07, extent[3]-0.05,str(numberstoplot[c]+1),fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
                    c+=1

        else:
            #if a figure/ax objects are passed, clear the axis and replot the maze
            if (ax is not None) and (fig is not None): 
                ax.clear()
                self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
            # else if they are not passed plot the maze
            if (fig, ax) == (None, None):
                fig, ax = self.plotMazeStructure(hist_id=hist_id)
            
            if number == None: number = random.randint(a=0,b=self.mazeAgent.stateSize-1)

            ax.imshow(gridFields[number],extent=extent,interpolation=None)

            if plotTimeStamp == True: 
                ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
            if show==False:
                plt.close(fig)
        
        saveFigure(fig, "gridField")
        return fig, ax
        
    def plotPlaceFields(self, hist_id=-1):
        fig, ax = self.plotMazeStructure(hist_id=hist_id)
        for (i, centre) in enumerate(self.mazeAgent.centres):
            ax.text(centre[0],centre[1],str(i),fontsize=3,horizontalalignment='center',verticalalignment='center')
            circle = matplotlib.patches.Ellipse((centre[0],centre[1]), 2*self.mazeAgent.sigma1, 2*self.mazeAgent.sigma1, alpha=0.5, facecolor= np.random.choice(['C0','C1','C2','C3','C4','C5']))
            ax .add_patch(circle)
        saveFigure(fig, "basis")
        return fig, ax 
    
    def plotHeatMap(self):
        posdata = np.stack(self.mazeAgent.history['pos'])
        bins = list(self.mazeAgent.discreteCoords.shape[:2])
        bins.reverse()
        print(bins)
        hist = np.histogram2d(posdata[:,0],posdata[:,1],bins=bins)[0]
        print(hist.T.shape)
        print(self.mazeAgent.extent)
        fig, ax = self.plotMazeStructure(hist_id=-1)
        ax.imshow(hist.T, extent=self.mazeAgent.extent)
        return fig, ax


    def animateField(self, number=0,field='place',interval=100):
        if field == 'place':
            fig, ax = self.plotPlaceField(hist_id=0,number=number,show=False)
            anim = FuncAnimation(fig, self.plotPlaceField, fargs=(None, fig, ax, number, False, True, True), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'grid':
            fig, ax = self.plotGridField(hist_id=0,number=number,show=False)
            anim = FuncAnimation(fig, self.plotGridField, fargs=(None, fig, ax, number, False, True, True), frames=len(self.snapshots), repeat=False, interval=interval)
        today = datetime.strftime(datetime.now(),'%y%m%d')
        now = datetime.strftime(datetime.now(),'%H%M')
        anim.save("./figures/animations/anim"+field+today+now+".mp4")

        return anim



wall_oneRoom = 
def perp(a=None):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0] 
    return b

def theta(segment):
    eps = 1e-6
    if segment.shape == (2,): 
        return np.mod(np.arctan2(segment[1],(segment[0] + eps)),2*np.pi)
    elif segment.shape == (2,2):
        return np.mod(np.arctan2((segment[1][1]-segment[0][1]),(segment[1][0] - segment[0][0] + eps)), 2*np.pi)


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


