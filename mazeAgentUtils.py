import numpy as np 

import pandas as pd 
from tqdm.notebook import tqdm
from datetime import datetime 
import numbers
from pprint import pprint as pprint
import os
import pickle
import scipy
from scipy.stats import vonmises
from scipy.spatial import distance_matrix
import time as Time 
import dill 
import cmath 
import sys 

sys.path.insert(-1,"../tomplotlib/")
from tomplotlib import *


#Default parameters for MazeAgent 
defaultParams = { 

          #Maze params 
          'mazeType'            : 'oneRoom',  #type of maze, define in getMaze() function
          'stateType'           : 'gaussian', #feature on which to TD learn (onehot, gaussian, gaussianCS, circles, bump)
          'movementPolicy'      : 'raudies',  #movement policy (raudies, random walk, windows screensaver)
          'roomSize'            : 1,          #maze size scaling parameter, metres
          'dt'                  : None,       #simulation time disretisation (defualts to largest )
          'dx'                  : 0.01,       #space discretisation (for plotting, movement is continuous)
          'speedScale'          : 0.16,       #movement speed scale, metres/second
          'rotSpeedScale'       : None,       #rotational speed scale, radians/second
          'initPos'             : [0.1,0.1],  #initial position [x0, y0], metres
          'initDir'             : [1,0],      #initial direction, unit vector
          'nCells'              : None,       #how many features to use
          'centres'             : None,       #array of receptive field positions. Overwrites nCells
          'sigma'               : 1,          #basis cell width scale (irrelevant for onehots)
          'doorsClosed'         : True,       #whether doors are opened or closed in multicompartment maze
          'reorderCells'        : True,       #whether to reorde the cell centres which have been provided
          'firingRateLookUp'    : False,      #use quantised lookup table for firing rates 
          'biasDoorCross'       : False,      #if True, in twoRoom maze door crossings are biased towards
          'biasWallFollow'      : True,       #if True, agent aligns to wall when gets too near.

          #TD params 
          'tau'                 : 4,          #TD decay time, seconds
          'TDdx'                : 0.01,       #rough distance between TD learning updates, metres 
          'alpha'               : 0.01,       #TD learning rate 
          'successorFeatureNorm': 100,        #linear scaling on successor feature definition found to improve learning stability
          'TDreg'               : 0.01,       #L2 regularisation 
          
          #STDP params
          'peakFiringRate'      : 5,          #peak firing rate of a cell (middle of place field, preferred theta phase)
          'tau_STDP_plus'       : 20e-3,      #pre trace decay time
          'tau_STDP_minus'      : 40e-3,      #post trace decay time
          'a_STDP'              : -0.4,       #pre-before-post potentiation factor (post-before-pre = 1) 
          'eta'                 : 0.05,       #STDP learning rate
          'baselineFiringRate'  : 0,          #baseline firing rate for cells 


          #Theta precession params
          'thetaFreq'           : 10,         #theta frequency
          'precessFraction'     : 0.5,        #fraction of 2pi the prefered phase moves through
          'kappa'               : 1,          # von mises spread parameter

}

class testclass():
    def printathing(self): print("New")

class MazeAgent():
    """MazeAgent defines an agent moving around a maze. 
    The agent moves according to a predefined movement policy
    As the agent moves it learns a successor representation over state vectors according to a TD learning rule 
    The movement polcy is 
        (i)  continuous in space. There is no discretisation of location. Time is discretised into steps of dt
        (ii) completely decoupled from the TD learning.
    TD learning is 
        (i)  state general. i.e. it learns generic SRs for feature vectors which are not necessarily onehot. See de Cothi and Barry, 2020  
        (ii) time continuous. Defined in terms of a memory decay time tau, not unitless gamma. Any two states can be used fro a TD learning step irrespective of their seperation in time. 
    As the rat moves and learns its position and time stamps are continually saved. Periodically a snapshot of the current SR matrix and state of other parameters in the maze are also saved. 
    """   
    def __init__(self,
                params={},
                loadFromFileCalled=None):
        """Sets the parameters of the maze anad agent (using default if not provided) 
        and initialises everything. This includes: 
        •initilising history dataframes
        •making the maze (a dictionary of "walls" which cant be crossed)
        •setting position, velocity, time
        •discretising space into coordinates for later plotting
        •initialising basis features (gaussian centres, fourier frequencies etc.)
        •initialising SR matrix 

        Args:
            params (dict, optional): A dictionary of parameters which you want to differ from the default. Defaults to {}.
        """        
        if loadFromFileCalled is not None: 
            self.loadFromFile(name=loadFromFileCalled)
            
        else:
            print("Setting parameters")
            for key, value in defaultParams.items():
                setattr(self, key, value)
            self.updateParams(params)

            print("Initialising")
            self.initialise()
            print("DONE")

    def updateParams(self,
                     params : dict):        
        """Updates parameters from a dictionary. 
        All parameters found in params will be updated to new value

        Args:
            params (dict): dictionary of parameters to change
            initialise (bool, optional): [description]. Defaults to False.
        """        
        for key, value in params.items():
            setattr(self, key, value)

    def initialise(self): #should only be called once at the start 
        """Initialises the maze and agent. Should only be called once at the start.
        """        
        #initialise history dataframes
        print("   making state/history dataframes")
        self.mazeState = {}
        self.history = pd.DataFrame(columns = ['t','pos','delta','runID']) 
        self.snapshots = pd.DataFrame(columns = ['t','M','W','mazeState'])

        #set pos/vel
        print("   initialising velocity, position and direction")
        self.pos = np.array(self.initPos)
        self.speed = self.speedScale
        self.dir = np.array(self.initDir)

        #time and runID
        print("   setting time/run counters")
        self.t = 0
        self.runID = 0  
        self.thetaPhase = self.thetaFreq*(self.t%(1/self.thetaFreq))*2*np.pi


        #make maze 
        print("   making the maze walls")
        self.walls = getWalls(mazeType=self.mazeType, roomSize=self.roomSize)
        walls = self.walls.copy()
        if self.doorsClosed == False: 
            del walls['doors']
            self.mazeState['walls'] = walls
        elif self.doorsClosed == True: 
            self.mazeState['walls'] = walls

        #extent, xArray, yArray, discreteCoords
        print("   discretising position for later plotting")
        if abs((self.roomSize / self.dx) - round(self.roomSize / self.dx)) > 0.00001:
            print("      dx must be an integer fraction of room size, setting it to %.4f, %g along room length" %(self.roomSize / round(self.roomSize / self.dx), round(self.roomSize / self.dx)))
            self.dx = self.roomSize / round(self.roomSize / self.dx)
        minx, maxx, miny, maxy = 0, 0, 0, 0
        for room in self.walls:
            wa = self.walls[room]
            minx, maxx, miny, maxy = min(minx,np.min(wa[...,0])), max(maxx,np.max(wa[...,0])), min(miny,np.min(wa[...,1])), max(maxy,np.max(wa[...,1])) 
        self.extent = np.array([minx,maxx,miny,maxy])
        self.width = maxx-minx
        self.height = maxy-miny
        self.xArray = np.arange(minx + self.dx/2, maxx, self.dx)
        self.yArray = np.arange(miny + self.dx/2, maxy, self.dx)[::-1]
        x_mesh, y_mesh = np.meshgrid(self.xArray,self.yArray)
        coordinate_mesh = np.array([x_mesh, y_mesh])
        self.discreteCoords = np.swapaxes(np.swapaxes(coordinate_mesh,0,1),1,2) #an array of discretised position coords over entire map extent 
        self.mazeState['extent'] = self.extent

        #handle None params
        print("   handling undefined parameters")
        if self.dt == None: 
            self.dt = min(self.tau_STDP_plus,self.tau_STDP_minus) / 2
        if self.pos is None: 
            ex = self.extent
            self.pos = np.array([ex[0] + 0.2*(ex[1]-ex[0]),ex[2] + 0.2*(ex[3]-ex[2])])
        if self.dir is None: 
            if self.mazeType == 'longCorridor': self.dir = np.array([0,1])
            elif self.mazeType == 'loop': self.dir = np.array([1,0])
            else: self.dir = np.array([1,1]) / np.sqrt(2)
        if self.rotSpeedScale is None: 
            if self.mazeType == 'loop' or self.mazeType == 'longCorridor':
                self.rotSpeedScale = np.pi
            else: 
                self.rotSpeedScale = 3*np.pi
        if (self.nCells is None) and (self.centres is None): 
            ex = self.extent
            area, pcarea  = (ex[1]-ex[0])*(ex[3]-ex[2]), np.pi * ((self.sigma/2)**2)
            cellsPerArea = 10
            self.nCells = int(cellsPerArea * area / pcarea) #~10 in any given place
        if self.mazeType == 'TMaze':
            self.LRDecisionPending=True
        self.doorPassage = False
        self.doorPassageTime = 0
        self.lastTurnUpdate = -1
        self.randomTurnSpeed = 0

        #initialise basis cells and M (successor matrix)
        print("   initialising basis features for learning")

        if self.stateType in ['gaussian', 'gaussianCS','gaussianThreshold', 'circles','onehot','bump']:
            if self.centres is not None: #if we don't provide locations for cell centres...
                self.nCells = self.centres.shape[0]
                self.stateSize = self.nCells
            else: #scatter some ourselves (making sure they aren't too close)
                self.stateSize=self.nCells
                xcentres = np.random.uniform(self.extent[0],self.extent[1],self.nCells)
                ycentres = np.random.uniform(self.extent[2],self.extent[3],self.nCells)
                self.centres = np.array([xcentres,ycentres]).T
                inds = self.centres[:,0].argsort()
                self.centres = self.centres[inds]
                print("   checking basis cells aren't too close") 
                min_d = 0.1/0.9
                done = False
                while done != True:
                    min_d *= 0.9
                    print("     min seperation distance:  %.1f cm" %(min_d*100))
                    count = 0
                    while count <= 10:
                        d = distance_matrix(self.centres,self.centres)
                        d  += 0.1*np.eye(d.shape[0])
                        d_xid, d_yid = np.where(d < min_d)
                        print('      ',int(len(d_xid)/2),' overlapping pairs',end='\n')
                        if len(d_xid) == 0:
                            done = True 
                            break
                        to_remove = []
                        for i in range(len(d_xid)):
                            if d_xid[i] < d_yid[i]:
                                to_remove.append(d_xid[i])
                        to_remove = np.unique(to_remove)
                        xcentres = np.random.uniform(self.extent[0],self.extent[1],len(to_remove))
                        ycentres = np.random.uniform(self.extent[2],self.extent[3],len(to_remove))
                        self.centres[to_remove] = np.array([xcentres,ycentres]).T
                        count += 1
            self.M = np.eye(self.stateSize)
            self.W = self.M.copy() / self.nCells
            self.M_theta = self.M.copy()
            self.W_notheta = self.W.copy()



            #order the place cells so successor matrix has some structure:
            if self.reorderCells==True:
                if self.mazeType == 'twoRooms': #from centre outwards
                    middle = np.array([self.extent[1]/2,self.extent[3]/2])
                    distance_to_centre = np.linalg.norm(middle - self.centres,axis=1)
                    distance_to_centre = distance_to_centre * (2*(self.centres[:,0]>middle[0])-1)
                    inds = distance_to_centre.argsort()
                    self.centres = self.centres[inds]
                else: #from left to right
                    inds = self.centres[:,0].argsort()
                    self.centres = self.centres[inds]

        elif self.stateType == 'fourier':
            self.stateSize = self.nCells
            self.kVectors = np.random.rand(self.nCells,2) - 0.5
            self.kVectors /= np.linalg.norm(self.kVectors, axis=1)[:,None]
            self.kFreq = 2*np.pi / np.random.uniform(0.01,1,size=(self.nCells))
            self.phi = np.random.uniform(0,2*np.pi,size=(self.nCells))
            self.M = np.eye(self.stateSize)
            #self.M = np.zeros((self.stateSize,self.stateSize))
        
        if hasattr(self.sigma,"__len__"):
            if self.sigma.__len__() == self.nCells:
                self.sigmas = self.sigma
        else:
            self.sigmas = np.array([self.sigma]*self.nCells)

        #array of states, one for each discretised position coordinate 
        print("   calculating state vector at all discretised positions")
        self.statesAlreadyInitialised = False
        self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords,stateType=self.stateType,verbose=True) #an array of discretised position coords over entire map extent 
        self.statesAlreadyInitialised = True

        #store time zero snapshot
        snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'W': [self.W.copy()],'W_notheta': [self.W_notheta.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = self.snapshots.append(snapshot)

        #STDP stuff
        print("   initialising STDP weight matrix and traces")
        self.preTrace = np.zeros(self.nCells) #causes potentiation 
        self.preTrace_notheta = np.zeros(self.nCells) #causes potentiation 
        self.postTrace = np.zeros(self.nCells) #causes depression 
        self.postTrace_notheta = np.zeros(self.nCells) #causes depression
        self.lastSpikeTime = np.array(-10.0)
        self.lastSpikeTime_notheta = np.array(-10.0)
        self.spikeCount = np.array(0)
        self.spikeCount_notheta = np.array(0)

    def runRat(self,
            trainTime=10,
            saveEvery=0.5,
            TDSRLearn=True,
            STDPLearn=True):
        """The main experiment call.
        A "run" consists of a period where the agent explores the maze according to the movement policy. 
        As it explores it learns, by TD, a successor representation over state vectors. 
        The can be called multiple times. Each successive run will be saved in self.history with an increasing runID
        Snapshots of the current SR matrix and mazeState can be saved along the way
        Runs can be interrupted with KeyboardInterrupt, data will still be saved. 
        Args:
            trainTime (int, optional): How long to explore in minutes. Defaults to 10.
            saveEvery (int, optional): Frequency to save snapshots, in minutes. Defaults to 1.
            TDSRLearn (bool,optional): toggles whether to do TD learning 
            STDPLearn (bool, optional): toggles whether to do STDP learning 
        """        
        steps = int(trainTime * 60 / self.dt) #number of steps to perform 

        hist_t = np.zeros(steps)
        hist_pos = np.zeros((steps,2))
        hist_delta = np.zeros(steps)

        lastTDstep, distanceToTD = 0, np.random.exponential(self.TDdx) #2cm scale
        
        """Main training loop. Principally on each iteration: 
            • always updates motion policy
            • often does TD learning step

            • sometimes saves snapshot"""
        for i in tqdm(range(steps)): #main training loop

            try:
                #update pos, velocity, direction and time according to movement policy
                self.movementPolicyUpdate()
                if i > 1:

                    # print(self.pos)
                    """STDP learning step"""
                    if (STDPLearn == True) and (self.stateType in  ['bump','gaussian', 'gaussianCS','gaussianThreshold', 'circles']):
                        _ = self.STDPLearningStep(dt = self.t - hist_t[i-1])

                            
                    """TD learning step"""
                    if TDSRLearn == True: 
                        
                        alpha = self.alpha
                        try: alpha_ = alpha[0] * np.exp(-(i/steps)*(np.log(self.alpha[0]/self.alpha[1]))) #decaying alpha
                        except: alpha_ = self.alpha
                        

                        if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= distanceToTD: #if it's moved over 2cm meters from last step 
                            dtTD = self.t - hist_t[lastTDstep]
                            delta = self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dtTD, tau=self.tau, alpha=alpha_)
                            lastTDstep = i 
                            distanceToTD = np.random.exponential(self.TDdx)
                            hist_delta[i] = delta



                self.thetaPhase = self.thetaFreq*(self.t%(1/self.thetaFreq))*2*np.pi #8Hz theta 

                #update history arrays
                hist_pos[i] = self.pos
                hist_t[i] = self.t

                #save snapshot 
                if (isinstance(saveEvery, numbers.Number)) and (i % int(saveEvery * 60 / self.dt) == 0):
                    snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta':[self.W_notheta.copy()], 'mazeState':[self.mazeState]})
                    self.snapshots = self.snapshots.append(snapshot)

            except KeyboardInterrupt: 
                print("Keyboard Interrupt:")
                break
            except ValueError as error:
                print("ValueError:")
                print(error)
                print(f"   Rat position: {self.pos}")
                break

        self.runID += 1
        runHistory = pd.DataFrame({'t':list(hist_t[:i]), 'pos':list(hist_pos[:i]),'delta':list(hist_delta[:i])})
        self.history = self.history.append(runHistory)
        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta':[self.W_notheta.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = self.snapshots.append(snapshot)

        #find and save grid/place cells so you don't have to repeatedly calculate them when plotting 
        print("Calculating place and grid cells")
        self.gridFields = self.getGridFields(self.M)
        self.placeFields = self.getPlaceFields(self.M)

        if TDSRLearn == True: 
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)
            delta = np.array(hist_delta)
            time = np.array(hist_t)
            time = time[delta!=0] / 60
            delta = delta[delta!=0]
            time, delta = time[::10], delta[::10]
            smooth_delta = [np.mean(delta[max(0,i-100):min(i+100,len(delta))]) for i in range(len(delta))]
            fig, ax = plt.subplots(figsize=(2,1))
            ax.scatter(time,delta,s=0.5,alpha=0.5)
            ax.scatter(time,smooth_delta,s=1,alpha=0.5,c='C2')
            ax.set_xlabel("Time / min")
            ax.set_ylabel("Update size")

    def TDLearningStep(self, pos, prevPos, dt, tau, alpha):
        """TD learning step
            Improves estimate of SR matrix, M, by a TD learning step. 
            By default this is done using learning rule for generic feature vectors (see de Cothi and Barry 2020). 
            If stateType is onehot, additional efficiencies can be gained by using onehot specific learning rule (see Stachenfeld et al. 2017)
            Does time continuous TD learning (see Doya, 2000)
        Args:
            pos: position at t+dt (t)
            prevPos (array): position at t (t-dt)
            dt (float): time difference between two positions
            tau (float or int): memory decay time (analogous to gamma in TD, gamma = 1 - dt/tau)
            alpha (float): learning rate
            mask (bool or str): whether to mask TM update to update only cells near current location
            asynchronus (bool): update cells asynchronusly (like hopfield)
        """
        state = self.posToState(pos,stateType=self.stateType) 
        prevState = self.posToState(prevPos,stateType=self.stateType) 

        data = ( (state,                        prevState,                    self.M        ) ,
                 (self.thetaModulation(state),  self.thetaModulation(state),  self.M_theta) )

        
        for i, (state, prevState, M) in  enumerate(data): 
            #onehot optimised TD learning 
            if self.stateType == 'onehot': 
                s_t = np.argwhere(prevState)[0][0]
                s_tplus1 = np.argwhere(state)[0][0]
                Delta = state + (tau / dt) * ((1 - dt/tau) * M[:,s_tplus1] - M[:,s_t])
                M[:,s_t] += alpha * Delta - 2 * alpha * self.TDreg * M[:,s_t]

            #normal TD learning 
            else:
                delta = ((tau * dt) / (tau + dt)) * self.successorFeatureNorm * prevState + M @ ((tau/(tau + dt))*state - prevState)
                Delta = np.outer(delta, prevState)
                M += alpha * Delta - 2 * alpha * self.TDreg * M #regularisation
            
            if i == 0: 
                Del = Delta

        return np.linalg.norm(Del)
        
    def STDPLearningStep(self,dt):       
        """Takes the curent theta phase and estimate firing rates for all basis cells according to a simple theta sweep model. 
           From here it samples spikes and performs STDP learning on a weight matrix.

        Args:
            dt (float): Time step length 

        Returns:
            float array: vector of firing rates for this time step 
        """   
        firingRate_CA3 = self.posToState(self.pos)

        data = (  (firingRate_CA3,    self.W_notheta,  self.preTrace_notheta,  self.postTrace_notheta,  self.lastSpikeTime_notheta, self.spikeCount_notheta),
                  (firingRate_CA3,    self.W,          self.preTrace,          self.postTrace,          self.lastSpikeTime        , self.spikeCount        ) )

        
        for i, (firingRate_CA3, W, preTrace, postTrace, lastSpikeTime, spikeCount) in enumerate(data): 
            
            firingRate_CA3_ = self.peakFiringRate * firingRate_CA3 + self.baselineFiringRate
            W_norm = W / np.sum(W,axis=1)[:np.newaxis]
            firingRate_CA1_ = np.maximum(np.matmul(W_norm,firingRate_CA3_),0) 

            if i == 1:
                firingRate_CA3_ =  self.thetaModulation(firingRate_CA3_)#scale firing rate and add noise                
                firingRate_CA1_ =  self.thetaModulation(firingRate_CA1_)#scale firing rate and add noise                

            n_spike_list_CA3 = np.random.poisson(firingRate_CA3_*dt)
            spikingNeurons_CA3 = (n_spike_list_CA3 != 0) #in short time dt cells can spike 0 or 1 time only (good enough approximation) 
            spikeCount += sum(spikingNeurons_CA3)
            spikeTimes_CA3 = np.random.uniform(self.t,self.t+dt,self.nCells)[spikingNeurons_CA3]
            spikeIDs_CA3 = np.arange(self.nCells)[spikingNeurons_CA3]
            area_CA3 = np.array([0]*len(spikeIDs_CA3))
            spikeList_CA3 = np.vstack((spikeIDs_CA3,spikeTimes_CA3,area_CA3)).T

            n_spike_list_CA1 = np.random.poisson(firingRate_CA1_*dt)
            spikingNeurons_CA1 = (n_spike_list_CA1 != 0) #in short time dt cells can spike 0 or 1 time only (good enough approximation) 
            spikeCount += sum(spikingNeurons_CA1)
            spikeTimes_CA1 = np.random.uniform(self.t,self.t+dt,self.nCells)[spikingNeurons_CA1]
            spikeIDs_CA1 = np.arange(self.nCells)[spikingNeurons_CA1]
            area_CA1 = np.array([1]*len(spikeIDs_CA1))
            spikeList_CA1 = np.vstack((spikeIDs_CA1,spikeTimes_CA1,area_CA1)).T


            spikeList = np.concatenate((spikeList_CA3,spikeList_CA1))

            spikeList = spikeList[np.argsort(spikeList[:,1])]   

            for spikeInfo in spikeList:
                cell, time, CA1 = int(spikeInfo[0]), spikeInfo[1], (spikeInfo[2]==1.0)

                timeDiff = time - lastSpikeTime 

                preTrace        *= np.exp(- timeDiff / self.tau_STDP_plus) #traces for all cells decay...
                postTrace       *= np.exp(- timeDiff / self.tau_STDP_minus) #traces for all cells decay...

                if CA1 == True:
                    #a post synpatic cell has fired, stregnthen all synapses into this cell according to presynaptic trace               
                    W[cell,:]       += self.eta * preTrace #weights to postsynaptic neuron (should increase when post fires)
                    postTrace[cell] += self.a_STDP  #update trace (post trace probably negative)
                elif CA1 == False: 
                    #a pre synpatic cell has fired, weaken all synapses out of this cell according to postsynaptic trace               
                    W[:,cell]       += self.eta * postTrace #weights to presynaptic neuron (should decrease when post fires) 
                    preTrace[cell]  += 1 #update trace 


                lastSpikeTime += timeDiff

            if i == 1: 
                thetaFiringRate = firingRate_CA3_

        return thetaFiringRate

    
    def thetaModulation(self, firingRate, position=None, direction=None):
        """Takes a firing rate vector and modulates it to account for theta phase precession

        Args:
            firingRate (np.array): The raw (position dependent) firing rate vector to be modulated 
            position (np.array(2,), optional): The agent position. Defaults to None.
            direction (np.array(2,), optional): The agent direction. Defaults to None.
        """        
        if position is None:
            position = self.pos
        if direction is None:
            direction = self.dir

        vectorToCells = self.vectorsToCellCentres(position)
        sigmasToCellMidline = (np.dot(vectorToCells,direction) / np.linalg.norm(direction))  / self.sigmas #as mutiple of sigma
        preferedThetaPhase = np.pi + sigmasToCellMidline * self.precessFraction * np.pi

        phaseDiff = preferedThetaPhase - self.thetaPhase
        modulatedFiringRate = firingRate * vonmises.pdf(phaseDiff,kappa=self.kappa) * 2*np.pi

        return modulatedFiringRate
        

    def movementPolicyUpdate(self):
        """Movement policy update. 
            In principle this does a very simple thing: 
            • updates time by dt, 
            • updates position along the velocity direction 
            • updates velocity (speed and direction) accoridng to a movement policy
            In reality it's a complex function as the policy requires checking for immediate or upcoming collisions with all walls at each step.
            This is done by function self.checkWallIntercepts()
            What it does with this info (bounce off wall, turn to follow wall, etc.) depends on policy. 
        """

        dt = self.dt
        self.t += dt
        proposedNewPos = self.pos + self.speed * self.dir * dt
        proposedStep = np.array([self.pos,proposedNewPos])

        if (self.biasDoorCross == True) and (self.mazeType == 'twoRooms'): 
            #if agent crosses into door zone there's its turn direction is biased to try and cross the door 
            #this is done by setting agents direction in the right direction and not changing it again until after it's crossed
            doorRegionSize = 1
            if self.doorPassage == False:
                #if step cross into door region
                if (np.linalg.norm(self.pos - np.array([self.roomSize,self.roomSize/2])) > doorRegionSize) and (np.linalg.norm(proposedNewPos - np.array([self.roomSize,self.roomSize/2])) < doorRegionSize) and (abs(self.pos[0] - self.roomSize) > 0.01):
                    if 100*np.random.uniform(0,1) < 50: #start a doorPassage
                        self.doorPassage = True
                        self.doorPassageTime = self.t
                        return
                    else: #ignore this 
                        pass
            if self.doorPassage == True: 
                if ((self.pos[0]<(self.roomSize)) != (proposedNewPos[0]<(self.roomSize))) or ((self.t - self.doorPassageTime)*self.speedScale > 2*doorRegionSize):
                    self.doorPassage = False
                    if ((self.pos[0]<(self.roomSize)) != (proposedNewPos[0]<(self.roomSize))): 
                        print("crossed",self.t)
                    if ((self.t - self.doorPassageTime)*self.speedScale > 2*doorRegionSize):
                        print("time")
        
        checkResult = self.checkWallIntercepts(proposedStep)
        if self.movementPolicy == 'randomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir,turnAngle=randomTurnSpeed*dt)
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'trueRandomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                self.dir = turn(self.dir,turnAngle=np.random.uniform(0,2*np.pi))
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'leftRightRandomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                self.dir = turn(self.dir,turnAngle=np.random.choice([0,np.pi]))
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'raudies':
            if checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
            elif ((checkResult[0] == 'collisionAhead') and (self.biasWallFollow==True)):
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'follow')
            elif (checkResult[0] == 'noImmediateCollision') or (((checkResult[0] == 'collisionAhead') and (self.biasWallFollow==False))):
                self.pos = proposedNewPos

            
            self.speed = np.random.rayleigh(self.speedScale)
            if self.t - self.lastTurnUpdate >= 0.1: #turn updating done at intervals independednt of dt or else many small turns cancel out but few big ones dont 
                randTurnMean = 0
                if self.doorPassage == True: 
                        d_theta  = theta(self.dir) - theta(np.array([self.roomSize,self.roomSize/2]) - self.pos)
                        if d_theta > 0: randTurnMean = -self.rotSpeedScale
                        else: randTurnMean = self.rotSpeedScale
                self.randomTurnSpeed = np.random.normal(randTurnMean,self.rotSpeedScale)
                self.lastTurnUpdate = self.t
            self.dir = turn(self.dir, turnAngle=self.randomTurnSpeed*dt)

        if self.movementPolicy == 'windowsScreensaver':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.mazeType == 'loop':
            self.pos[0] = self.pos[0] % self.roomSize
        
        if self.mazeType == 'TMaze':
            if (self.pos[0] > self.roomSize+0.05) and (self.LRDecisionPending==True):
                if np.random.choice([0,1],p=[0.66,0.34]) == 0:
                    self.dir = np.array([0,1])
                else:
                    self.dir = np.array([0,-1])
                self.LRDecisionPending=False
            if self.pos[1] > self.extent[3] or self.pos[1] < self.extent[2]:
                self.pos = np.array([0,1])
                self.dir = np.array([1,0])
                self.LRDecisionPending=True


        #catchall instances a rat escapes the maze by accident, pops it 2cm within maze 
        if ((self.pos[0] < self.extent[0]) or 
            (self.pos[0] > self.extent[1]) or 
            (self.pos[1] < self.extent[2]) or 
            (self.pos[1] > self.extent[3])):
            print(self.pos)
            self.pos[0] = max(self.pos[0],self.extent[0]+0.02)
            self.pos[0] = min(self.pos[0],self.extent[1]-0.02)
            self.pos[1] = max(self.pos[1],self.extent[2]+0.02)
            self.pos[1] = min(self.pos[1],self.extent[3]-0.02)
            print("Rat escaped!")
            if self.mazeType == 'TMaze':
                self.dir=np.array([1,0])
                self.LRDecisionPending = True
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)

    def vectorsToCellCentres(self,pos,distance=False):
        """Takes a posisiton vector shape (2,) and returns an array of shape (nCells,2) of the 
        shortest vector path to all cells, taking into account loop geometry etc. 

        Args:
            pos (array): position vector shape (2,)

        Returns:
            vectorToCells (array): shape (30,2)
        """        
        if self.mazeType == 'loop' and self.doorsClosed == False:
            pos_plus = pos + np.array([self.roomSize,0])
            pos_minus = pos - np.array([self.roomSize,0])
            positions = np.array([pos,pos_plus,pos_minus])
            vectors = self.centres[:,np.newaxis,:] - positions[np.newaxis,:,:]
            shortest = np.argmin(np.linalg.norm(vectors,axis=-1),axis=1)
            shortest_vectors = np.diagonal(vectors[:,shortest,:],axis1=0,axis2=1).T

        else:
            shortest_vectors = self.centres - self.pos
            
        return shortest_vectors

    def distanceToCellCentres(self, pos):
        """Calculates distance to cell centres. 
           In the case of the two room maze, this distance is the shortest feasible walk carefully accounting for doorways etc. 
        Args:
            pos (no.array): The position to calculate the distances from 

        Returns:
            np.array: (nCells,) array of distances
        """         

        if self.mazeType == 'twoRooms': 
            distances = np.zeros(self.nCells)
            wall_x = self.walls['doors'][0][0][0]
            wall_y1, wall_y2 = self.walls['doors'][0][0][1], self.walls['doors'][0][1][1]
            for i in range(self.nCells):
                vec = np.array(pos - self.centres[i])
                if ((self.centres[i][0] < wall_x) and (pos[0] < wall_x)) or ((self.centres[i][0] > wall_x) and (pos[0] > wall_x)):
                    distances[i] = np.linalg.norm(vec)
                else: #cell and position in different rooms 
                    if self.doorsClosed == True:
                        distances[i] = 100*self.roomSize 
                        print(doorsClosed)
                    else:
                        step = np.array([pos,self.centres[i]])
                        if self.checkWallIntercepts(step)[0] == 'collisionNow':
                            pastBottomWall = np.linalg.norm(np.array([wall_x,wall_y1]) - pos) + np.linalg.norm(np.array([wall_x,wall_y1]) - self.centres[i])
                            pastTopWall = np.linalg.norm(np.array([wall_x,wall_y2]) - pos) + np.linalg.norm(np.array([wall_x,wall_y2]) - self.centres[i])
                            distances[i] = min(pastBottomWall,pastTopWall)
                        else: 
                            distances[i] = np.linalg.norm(vec)

        else: 
            shortest_vector = self.vectorsToCellCentres(pos)
            distances = np.linalg.norm(shortest_vector,axis=1)

        return distances


    def toggleDoors(self, doorsClosed = None): #this function could be made more advanced to toggle more maze options
        """Opens or closes door and updates mazeState
            mazeState stores the most recent version of the maze walls dictionary which will include 'door' wall only if doorsClosed is True
        Args:
            doorsClosed ([bool], optional): True is doors to be closed, False if doors to be opened. Defaults to None, in which case current door state is flipped.
        Returns:
            [dict]: the walls dictionary
        """        
        if doorsClosed is not None: 
            self.doorsClosed = doorsClosed
        else: self.doorsClosed = not self.doorsClosed

        walls = self.walls.copy()
        if self.doorsClosed == False: 
            del walls['doors']
            self.mazeState['walls'] = walls
        elif self.doorsClosed == True: 
            self.mazeState['walls'] = walls

        self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 

        return self.mazeState['walls']

    def checkWallIntercepts(self,proposedStep,collisionDistance=0.1): #proposedStep = [pos,proposedNextPos]
        """Given the cuurent proposed step [currentPos, nextPos] it calculates whether a collision with any of the walls exists along this step.
        There are three possibilities from most worrying to least:
            • there is a collision ON the current step. Do something immediately.
            • there is a collision along the current trajectory in the next few cm's, but not on the current step. Consider doing something.
            • there is no collision coming up soon. Carry on as you are. 
        Args:
            proposedStep (array): The proposed step. np.array( [ [x_current, y_current] , [x_next, y_next] ] )

        Returns:
            tuple: (str, array), (<whether there is no collision, collision now or collision ahead> , <the wall in question>)
        """        
        s1, s2 = np.array(proposedStep[0]), np.array(proposedStep[1])
        pos = s1
        ds = s2 - s1
        stepLength = np.linalg.norm(ds)
        ds_perp = perp(ds)

        collisionList = [[],[]]
        futureCollisionList = [[],[]]

        #check if the current step results in a collision 
        walls = self.mazeState['walls'] #current wall state

        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                w1, w2 = np.array(wall[0]), np.array(wall[1])
                dw = w2 - w1
                dw_perp = perp(dw)

                # calculates point of intercept between the line passing along the current step direction and the lines passing along the walls,
                # if this intercept lies on the current step and on the current wall (0 < lam_s < 1, 0 < lam_w < 1) this implies a "collision" 
                # if it lies ahead of the current step and on the current wall (lam_s > 1, 0 < lam_w < 1) then we should "veer" away from this wall
                # this occurs iff the solution to s1 + lam_s*(s2-s1) = w1 + lam_w*(w2 - w1) satisfies 0 <= lam_s & lam_w <= 1
                with np.errstate(divide='ignore'):
                    lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / (np.dot(ds, dw_perp))
                    lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / (np.dot(dw, ds_perp))

                #there are two situations we need to worry about: 
                # • 0 < lam_s < 1 and 0 < lam_w < 1: the collision is ON the current proposed step . Do something immediately.
                # • lam_s > 1     and 0 < lam_w < 1: the collision is on the current trajectory, some time in the future. Maybe do something. 
                if (0 <= lam_s <= 1) and (0 <= lam_w <= 1):
                    collisionList[0].append(wall)
                    collisionList[1].append([lam_s,lam_w])
                    continue

                if (lam_s > 1) and (0 <= lam_w <= 1):
                    if lam_s * stepLength <= collisionDistance: #if the future collision is under collisionDistance away
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

    def getPlaceFields(self, M=None, threshold=None):
        """Calculates receptive fiels of all place cells 
            There is one place cell for each feature cell. 
            A place cell (as  in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination is a row of the SR matrix. 
        Args:
            M (array): SR matrix
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """        
        if M is None: 
            M = self.M
        M = M.copy()
        #normalise: 
        # M = M / np.diag(M)[:,np.newaxis]
        if threshold is None: 
            placeCellThreshold =  0.9  #place cell threshold value (fraction of its maximum)
        else: 
            placeCellThreshold =  threshold
        placeFields = np.einsum("ij,klj->ikl",M,self.discreteStates)
        threshold = placeCellThreshold*np.amax(placeFields,axis=(1,2))[:,None,None]
        # threshold = placeCellThreshold
        placeFields = np.maximum(0,placeFields - threshold)
        return placeFields

    def getGridFields(self, M, alignToFinal=False):
        """Calculates receptive fiels of all grid cells 
            There is an equal number of grid cells as place cells and feature cells. 
            A grid cell (as in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination weights are the eigenvectors of the SR matrix. 
        Args:
            M (array): SR matrix
            alignToFinal (bool): Since negative of eigenvec is also eigenvec try maximise overlap with final one (for making animations)
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """
        M = M.copy()
        _, eigvecs = np.linalg.eig(M) #"v[:,i] is the eigenvector corresponding to the eigenvalue w[i]"
        eigvecs = np.real(eigvecs)
        gridCellThreshold = 0 
        gridFields = np.einsum("ij,kli->jkl",eigvecs,self.discreteStates)
        threshold = gridCellThreshold*np.amax(gridFields,axis=(1,2))[:,None,None]
        if alignToFinal == True:
            grids_final_flat = np.reshape(self.gridFields,(self.stateSize,-1))
            grids_flat = np.reshape(gridFields,(self.stateSize,-1))
            dotprods = np.empty(grids_flat.shape[0])
            for i in range(len(dotprods)):
                dotprodsigns = np.sign(np.diag(np.matmul(grids_final_flat,grids_flat.T)))
                gridFields *= dotprodsigns[:,None,None]
        gridFields = np.maximum(0,gridFields)
        return gridFields
    
    def posToState(self, pos, stateType=None, normalise=True, cheapNormalise=False,initialisingCells=False): #pos is an [n1, n2, n3, ...., 2] array of 2D positions
        
        if (self.statesAlreadyInitialised == False) or (self.firingRateLookUp == False):
            #calculates the firing rate of all cells 
            pos = np.array(pos)
            if stateType == None: stateType = self.stateType
        
            vector_to_cells = self.centres - pos
            distance_to_cells = [np.linalg.norm(vector_to_cells,axis=1)]
            closest_cell_ID = np.argmin(distance_to_cells)

            if (self.mazeType == 'loop') and (self.doorsClosed == False):
                distance_to_cells.append(np.linalg.norm(self.centres - pos + [self.extent[1],0],axis=1))
                distance_to_cells.append(np.linalg.norm(self.centres - pos - [self.extent[1],0],axis=1))
            
            if (self.mazeType == 'twoRooms'): 
                distance_to_cells = [self.distanceToCellCentres(pos)]

            if stateType == 'onehot':
                state = np.zeros(self.nCells)
                state[closest_cell_ID] = 1
            
            if stateType == 'gaussianThreshold':
                state = np.zeros(self.nCells)
                for distance in distance_to_cells: 
                    state += np.maximum(np.exp(-distance**2 / (2*(self.sigmas**2))) - np.exp(-1/2) , 0) / (1-np.exp(-1/2))
                    # state = state / (self.sigmas) #normalises so same no. spikes emitted for all cell sizes

            if stateType == 'gaussian':
                state = np.zeros(self.nCells)
                for distance in distance_to_cells: 
                    state += np.exp(-distance**2 / (2*(self.sigmas**2)))
                    state = state/self.sigmas

            if stateType == 'bump':
                state = np.zeros(self.nCells)
                for distance in distance_to_cells:
                    state[distance<self.sigmas] += np.e * np.exp(-1/(1-(distance/self.sigmas)**2))[distance<self.sigmas]
                    state[distance>=self.sigmas] += 0
                    state = state/self.sigmas
        
        else:
            #uses look up table to rapidly pull out the firing rates without having to recalculate them 

            closestQuantisedArea = np.unravel_index(np.argmin(np.linalg.norm(self.discreteCoords - pos,axis=-1)),self.discreteCoords.shape[:-1])
            state = self.discreteStates[closestQuantisedArea]


        return state


    def positionArray_to_stateArray(self, positionArray, stateType=None,verbose=False): 
        """Takes an array of 2D positions of size (n1, n2, n3, ..., 2)
        returns the state vector for each of these positions of size (n1, n2, n3, ..., N) where N is the size of the state vector
        Args:
            positionArray ([type]): [description]
            stateType ([type], optional): [description]. Defaults to None.
        """        
        if stateType == None: stateType = self.stateType
        states = np.zeros(positionArray.shape[:-1] + (self.nCells,))
        if verbose == False:
            for idx in np.ndindex(positionArray.shape[:-1]):
                states[idx] = self.posToState(pos = positionArray[idx],stateType = stateType)
        else: 
            for idx in tqdm(np.ndindex(positionArray.shape[:-1]),total=int(positionArray.size/2)):
                states[idx] = self.posToState(pos = positionArray[idx],stateType = stateType)

        return states
    
    def averageM(self, M=None):
        if M == None:
            M = self.M
        M_copy = M.copy()
        roll = int(self.nCells/2)
        for i in range(agent.nCells):
            M_copy[i,:] = np.roll(M[i,:],-i+roll)
        M_av,M_std = np.mean(M_copy,axis=0),np.std(M_copy,axis=0)
        M_av, M_std = M_av/np.max(M_av), M_std/np.max(M_std)
        return M_av, M_std
    
    def getMetrics(self,time=None):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time).abs().to_numpy().argmin()
            snapshot = self.snapshots.iloc[hist_id]
        else:
            snapshot = self.snapshots.iloc[-1]

        x = self.centres[:,0].copy()

        W = rowAlignMatrix(snapshot['W'].copy())
        W_notheta = rowAlignMatrix(snapshot['W_notheta'].copy())
        M = rowAlignMatrix(self.snapshots.iloc[-1]['M'].copy())

        mid = int(self.nCells / 2)

        #R2s
        R_W = Rsquared(W,M)              
        R_Wnotheta = Rsquared(W_notheta,M)  

        #SNRs
        SNR_W = (np.max(np.mean(W,axis=0)) - np.min(np.mean(W,axis=0))) / np.mean(np.std(W,axis=0)[mid-5:mid+5])
        SNR_Wnotheta = (np.max(np.mean(W_notheta,axis=0)) - np.min(np.mean(W_notheta,axis=0))) / np.mean(np.std(W_notheta,axis=0)[mid-5:mid+5])

        #skews
        W_flat = np.mean(W,axis=0)/np.trapz(np.mean(W,axis=0),x)
        W_flat /= np.max(W_flat)
        Wnotheta_flat = np.mean(W_notheta,axis=0)/np.trapz(np.mean(W_notheta,axis=0),x)
        Wnotheta_flat /= np.max(Wnotheta_flat)
        M_flat = np.mean(M,axis=0)/np.trapz(np.mean(M,axis=0),x)
        M_flat /= np.max(M_flat)
        try: skew_W = getSkewness(W_flat)
        except RuntimeError: skew_W = np.NaN
        try: skew_Wnotheta = getSkewness(Wnotheta_flat)
        except RuntimeError: skew_Wnotheta = np.NaN
        try: skew_M = getSkewness(M_flat)
        except RuntimeError: skew_M = np.NaN

        #peaks 
        peak_W = x[np.argmax(W_flat)] - 2.5
        peak_Wnotheta = x[np.argmax(Wnotheta_flat)] - 2.5
        peak_M = x[np.argmax(M_flat)] - 2.5
        return R_W, R_Wnotheta, SNR_W, SNR_Wnotheta, float(skew_W), float(skew_Wnotheta), float(skew_M), peak_W, peak_Wnotheta, peak_M

    def saveToFile(self,name,directory="../savedObjects/"):
        np.savez(directory+name+".npz",self.__dict__)
        return

    def loadFromFile(self,name,directory="../savedObjects/"):
        attributes_dictionary = np.load(directory+name+".npz",allow_pickle=True)['arr_0'].item()
        print("Loading attributes...",end="")
        for key, value in attributes_dictionary.items():
            setattr(self, key, value)
        print("done. use 'agent3.__dict__.keys()'  to see avaiable attributes")

    
        
def getWalls(mazeType, roomSize=1):
    """Stores and returns dictionaries containing all the walls of a maze
    Args:
        mazeType (str): Name of the maze 
        roomSize (int, optional): scaling parameter for roomsize. Defaults to 1 metre.
    Returns:
        dict: wall dictionary
    """    
    walls = {}
    rs = roomSize
    if mazeType == 'oneRoom':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0]],
                                [[rs,0],[0,0]]])
    elif mazeType == 'twoRooms':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0.6*rs]],
                                [[rs,0.4*rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[rs,0.4*rs]],
                                [[rs,0.6*rs],[rs,rs]],
                                [[rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,0]],
                                [[2*rs,0],[rs,0]]])
        walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]]])
    elif mazeType == 'fourRooms':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[0.4*rs,rs]],
                                [[0.6*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,0.6*rs]],
                                [[rs,0.4*rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[rs,0.4*rs]],
                                [[rs,0.6*rs],[rs,rs]],
                                [[rs,rs],[1.4*rs,rs]],
                                [[1.6*rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,0]],
                                [[2*rs,0],[rs,0]]])
        walls['room3'] = np.array([
                                [[0,rs],[0.4*rs,rs]],
                                [[0.6*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,1.4*rs]],
                                [[rs,1.6*rs],[rs,2*rs]],
                                [[rs,2*rs],[0,2*rs]],
                                [[0,2*rs],[0,rs]]])
        walls['room4'] = np.array([
                                [[rs,rs],[1.4*rs,rs]],
                                [[1.6*rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,2*rs]],
                                [[2*rs,2*rs],[rs,2*rs]],
                                [[rs,2*rs],[rs,1.6*rs]],
                                [[rs,1.4*rs],[rs,rs]]])
        walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]],
                                        [[0.4*rs,rs],[0.6*rs,rs]],
                                        [[rs,1.4*rs],[rs,1.6*rs]],
                                        [[1.4*rs,rs],[1.6*rs,rs]]])
    elif mazeType == 'twoRoomPassage':
        walls['room1'] = np.array([
                                [[0,0],[rs,0]],
                                [[rs,0],[rs,rs]],
                                [[rs,rs],[0.75*rs,rs]],
                                [[0.25*rs,rs],[0,rs]],
                                [[0,rs],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[2*rs,0]],
                                [[2*rs,0],[2*rs,rs]],
                                [[2*rs,rs],[1.75*rs,rs]],
                                [[1.25*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,0]]])
        walls['room3'] = np.array([
                                [[0,rs],[0,1.4*rs]],
                                [[0,1.4*rs],[2*rs,1.4*rs]],
                                [[2*rs,1.4*rs],[2*rs,rs]]])
        walls['doors'] = np.array([[[0.25*rs,rs],[0.75*rs,rs]],
                                [[1.25*rs,rs],[1.75*rs,rs]]])
    elif mazeType == 'longCorridor':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['longbarrier'] = np.array([
                                [[0.1*rs,0],[0.1*rs,0.9*rs]],
                                [[0.2*rs,rs],[0.2*rs,0.1*rs]],
                                [[0.3*rs,0],[0.3*rs,0.9*rs]],
                                [[0.4*rs,rs],[0.4*rs,0.1*rs]],
                                [[0.5*rs,0],[0.5*rs,0.9*rs]],
                                [[0.6*rs,rs],[0.6*rs,0.1*rs]],
                                [[0.7*rs,0],[0.7*rs,0.9*rs]],
                                [[0.8*rs,rs],[0.8*rs,0.1*rs]],
                                [[0.9*rs,0],[0.9*rs,0.9*rs]]])
    elif mazeType == 'rectangleRoom':
        ratio = np.pi/2.8
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[ratio*rs,rs]],
                                [[ratio*rs,rs],[ratio*rs,0]],
                                [[ratio*rs,0],[0,0]]])
    elif mazeType == 'loop':
        height = 0.2
        walls['room'] = np.array([
                                [[0,0],[rs,0]],
                                [[0,height],[rs,height]]])
        walls['doors'] = np.array([
                                [[0,0],[0,height]],
                                [[rs,0],[rs,height]]])
    
    elif mazeType == 'TMaze':
        walls['corridors'] = np.array([
                                [[0,0.05+rs],[rs,0.05+rs]],
                                [[0,-0.05+rs],[rs,-0.05+rs]],
                                [[rs,0.05+rs],[rs,rs+rs]],
                                [[rs+0.1,rs+rs],[rs+0.1,-rs+rs]],
                                [[rs,-0.05+rs],[rs,-rs+rs]]]) 
        walls['doors'] = np.array([[[rs,-0.05+rs-0.5],[rs+0.1,-0.05+rs-0.5]]])                               

    return walls

#MOVEMENT FUNCTIONS
def wallBounceOrFollow(currentDirection,wall,whatToDo='bounce'):
    """Given current direction, and wall and an instruction returns a new direction which is the result of implementing that instruction on the current direction
        wrt the wall. e.g. 'bounce' returns direction after elastic bounce off wall. 'follow' returns direction parallel to wall (closest to current heading)
    Args:
        currentDirection (array): the current direction vector
        wall (array): start and end coordinates of the wall
        whatToDo (str, optional): 'bounce' or 'follow'. Defaults to 'bounce'.
    Returns:
        array: new direction
    """    
    if whatToDo == 'bounce':
        wallPerp = perp(wall[1] - wall[0])
        if np.dot(wallPerp,currentDirection) <= 0:
            wallPerp = -wallPerp #it is now the perpendicular with smallest angle to dir 
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,currentDirection) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar, wallPerp = wallPar/np.linalg.norm(wallPar), wallPerp/np.linalg.norm(wallPerp) #normalise
        dir_ = wallPar * np.dot(currentDirection,wallPar) - wallPerp * np.dot(currentDirection,wallPerp)
        newDir = dir_/np.linalg.norm(dir_)
    elif whatToDo == 'follow':
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,currentDirection) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar = wallPar/np.linalg.norm(wallPar)
        dir_ = wallPar * np.dot(currentDirection,wallPar)
        newDir = dir_/np.linalg.norm(dir_)
    return newDir

def turn(currentDirection, turnAngle):
    """Turns the current direction by an amount theta, modulus 2pi
    Args:
        currentDirection (array): current direction 2-vector
        turnAngle (float): angle ot turn in radians
    Returns:
        array: new direction
    """    
    theta_ = theta(currentDirection)
    theta_ += turnAngle
    theta_ = np.mod(theta_, 2*np.pi)
    newDirection = np.array([np.cos(theta_),np.sin(theta_)])
    return newDirection

def perp(a=None):
    """Given 2-vector, a, returns its perpendicular
    Args:
        a (array, optional): 2-vector direction. Defaults to None.
    Returns:
        array: perpendicular to a
    """    
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0] 
    return b

def theta(segment):
    """Given a 'segment' (either 2x2 start and end positions or 2x1 direction bearing) 
         returns the 'angle' of this segment modulo 2pi
    Args:
        segment (array): The segment, (2,2) or (2,) array 
    Returns:
        float: angle of segment
    """    
    eps = 1e-6
    if segment.shape == (2,): 
        return np.mod(np.arctan2(segment[1],(segment[0] + eps)),2*np.pi)
    elif segment.shape == (2,2):
        return np.mod(np.arctan2((segment[1][1]-segment[0][1]),(segment[1][0] - segment[0][0] + eps)), 2*np.pi)

class Visualiser():
    def __init__(self, mazeAgent):
        self.mazeAgent = mazeAgent
        self.snapshots = mazeAgent.snapshots
        self.history = mazeAgent.history

    def plotMazeStructure(self,fig=None,ax=None,hist_id=-1,save=False):
        snapshot = self.snapshots.iloc[hist_id]
        extent, walls = snapshot['mazeState']['extent'], snapshot['mazeState']['walls']
        if (fig, ax) == (None, None): 
            fig, ax = plt.subplots(figsize=(4*(extent[1]-extent[0]),4*(extent[3]-extent[2])))
        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=8)
            # ax.set_xlim(left=extent[0]-0.05,right=extent[1]+0.05)
            # ax.set_ylim(bottom=extent[2]-0.05,top=extent[3]+0.05)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        if save == True: 
            saveFigure(fig, 'mazeStructure')
        return fig, ax
    
    def plotTrajectory(self,fig=None, ax=None, hist_id=-1,starttime=0,endtime=2):
        skiprate = max(1,int(0.015/(self.mazeAgent.speedScale * self.mazeAgent.dt)))
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        startid = self.history['t'].sub(starttime*60).abs().to_numpy().argmin()
        endid = self.history['t'].sub(endtime*60).abs().to_numpy().argmin()
        trajectory = np.stack(self.history['pos'][startid:endid])[::skiprate]
        ax.scatter(trajectory[:,0],trajectory[:,1],s=10,alpha=0.7,zorder=2)
        saveFigure(fig, "trajectory")
        return fig, ax

    
    def plotM(self,hist_id=-1, time=None, M=None,fig=None,ax=None,save=True,savename="",title="",show=True,plotTimeStamp=False,colorbar=True,whichM='M',colormatchto='TD_M'):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
        snapshot = self.snapshots.iloc[hist_id]
        if (ax is not None) and (fig is not None): 
            ax.clear()
        else:
            fig, ax = plt.subplots(figsize=(2,2))
        if M is None:
            if whichM == 'M': M = snapshot['M'].copy()
            elif whichM == 'W': M = snapshot['W'].copy()
            elif whichM == 'M_theta': M = self.mazeAgent.M_theta.copy()
            elif whichM == 'W_notheta': M = self.mazeAgent.W_notheta.copy()

        t = int(np.round(snapshot['t']))
        most_positive = np.max(M)
        most_negative = np.min(M)

        if colormatchto == 'TD_M': 
            M_colormatch = self.mazeAgent.M
        elif colormatchto == 'W_onDiag': 
            M_colormatch = self.mazeAgent.W_onDiag
        # elif colormatchto is not None:
        #     M_colormatch = np.load(colormatchto)
        else: 
            M_colormatch = np.array([-1,1])

        # if np.min(M)/np.min(M_colormatch) > np.max(M)/np.max(M_colormatch):
            # M *= np.min(M_colormatch)/np.min(M)
        # else:
        M_ = M.copy()
        # np.fill_diagonal(M_,0)
        non_diag_max = np.max(M_)
        non_diag_kind_max = np.mean(M_[M_>0.9*non_diag_max])
        M *= np.max(M_colormatch)/non_diag_kind_max

        im = ax.imshow(M,cmap='viridis',vmin=np.min(M_colormatch),vmax=np.max(M_colormatch))
        divider = make_axes_locatable(ax)
        try: cax.clear()
        except: 
            pass
        if colorbar == True:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im, cax=cax, ticks=[0])
            cb.outline.set_visible(False)

        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        ax.set_title(title)
        if save==True:
            saveFigure(fig, "M"+savename)
        if plotTimeStamp == True: 
            ax.text(100, 5,"%g s"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        
        try:
            return fig, ax, cb, cax
        except:
            return fig, ax 

    def addTimestamp(self, fig, ax, i=-1):
        t = self.mazeAgent.saveHist[i]['t']
        ax.text(x=0, y=0, t="%.2f" %t)

    def plotPlaceField(self, M=None, hist_id=-1, time=None, fig=None, ax=None, number=None, show=True, animationCall=False, plotTimeStamp=False,save=True,STDP=False,threshold=None,fitEllipse_=False):
        
        if M is None: 
            #time in minutes
            if time is not None: 
                hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
            #if a figure/ax objects are passed, clear the axis and replot the maze
            if (ax is not None) and (fig is not None): 
                ax.clear()
                self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
            # else if they are not passed plot the maze
            if (fig, ax) == (None, None):
                fig, ax = self.plotMazeStructure(hist_id=hist_id)
            
            if number == None: number = np.random.randint(0,self.mazeAgent.stateSize-1)
            
            snapshot = self.snapshots.iloc[hist_id]
            M = snapshot['M']
            if STDP==True: 
                M = snapshot['W']
            t = int(np.round(snapshot['t'] / 60))
        else:
            snapshot = self.snapshots.iloc[hist_id]
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        extent = snapshot['mazeState']['extent']
        placeFields = self.mazeAgent.getPlaceFields(M=M,threshold=threshold)
        ax.imshow(placeFields[number],extent=extent,interpolation=None)

        if fitEllipse_ == True: 
            (X,Y,Z),_ = fitEllipse(placeFields[number],coords=self.mazeAgent.discreteCoords)
            ax.contour(X, Y, Z, levels=[1], colors=('w'), linewidths=2, linestyles="dashed")

        
        if self.mazeAgent.mazeType == 'loop':
            x = self.mazeAgent.discreteCoords[10,:,0]

        peakid= np.argmax(placeFields[number])
        peakcoord = self.mazeAgent.discreteCoords.reshape(-1,2)[peakid]
        ax.scatter(peakcoord[0],peakcoord[1],marker='x',s=130,color='darkgrey',linewidth=4,edgecolors='darkgrey',alpha=1)

        if plotTimeStamp == True: 
            ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        if save==True:
            saveFigure(fig, "placeField")
        return fig, ax
    

    def plotReceptiveField(self, number=None, hist_id=-1, fig=None, ax=None, show=True, fitEllipse_=False):
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        if number == None: number = np.random.randint(0,self.mazeAgent.nCells-1)
        extent = self.mazeAgent.extent
        rf = self.mazeAgent.discreteStates[..., number]
        ax.imshow(rf,extent=extent,interpolation=None)
        peakid= np.argmax(rf)
        peakcoord = self.mazeAgent.discreteCoords.reshape(-1,2)[peakid]
        ax.scatter(peakcoord[0],peakcoord[1],marker='x',s=130,color='darkgrey',linewidth=4,edgecolors='darkgrey',alpha=1)
        if self.mazeAgent.mazeType == 'loop':
            x = self.mazeAgent.discreteCoords[10,:,0]
        if fitEllipse_ == True: 
            (X,Y,Z),_= fitEllipse(rf,coords=self.mazeAgent.discreteCoords)
            ax.contour(X, Y, Z, levels=[1], colors=('w'), linewidths=2, linestyles="dashed")
        if show==False:
            plt.close(fig)
        saveFigure(fig, "receptiveField")
        return fig, ax
    

    def plotGridField(self, hist_id=-1, time=None, fig=None, ax=None, number=0, show=True, animationCall=False, plotTimeStamp=False,save=True,STDP=False):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        if STDP==True: 
            M = snapshot['W'].T    
        t = snapshot['t'] / 60
        extent = snapshot['mazeState']['extent']
        if hist_id == -1 and animationCall == False:
            gridFields = self.mazeAgent.gridFields
        else:
            gridFields = self.mazeAgent.getGridFields(M=M,alignToFinal=True)

        def sigmoid(x):
            return np.exp(x) / (np.exp(x) + np.exp(-x))
        if number == 'many': 
            fig = plt.figure(figsize=(10, 10*((extent[3]-extent[2])/(extent[1]-extent[0]))))
            gs = matplotlib.gridspec.GridSpec(6, 6, hspace=0.1, wspace=0.1)
            c=0
            # numberstoplot = np.array([60 + 5*i for i in np.arange(36)])
            numberstoplot = np.concatenate((np.array([0,1,2,3,4,5]),np.geomspace(6,gridFields.shape[0]-1,30).astype(int)))
            for i in range(6):
                for j in range(6):
                    ax = plt.subplot(gs[i,j])
                    # ax.imshow(sigmoid(gridFields[numberstoplot[c]]),extent=extent,interpolation=None)
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
            
            if number == None: number = np.random.randint(a=0,b=self.mazeAgent.stateSize-1)

            ax.imshow(gridFields[number],extent=extent,interpolation=None)

            if plotTimeStamp == True: 
                ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
            if show==False:
                plt.close(fig)
        
        if save==True:
            saveFigure(fig, "gridField")
        return fig, ax
        
    def plotFeatureCells(self, hist_id=-1,textlabel=True,shufflebeforeplot=False,centresOnly=False,onepink=False,threetypes=False):
        fig, ax = self.plotMazeStructure(hist_id=hist_id)
        centres = self.mazeAgent.centres.copy()
        ids = np.arange(len(centres))
        if shufflebeforeplot==True:
            np.random.shuffle(ids)
        centres = centres[ids]
        for (i, centre) in enumerate(centres):
            # if i%10==0:
                if textlabel==True:
                    ax.text(centre[0],centre[1],str(ids[i]),fontsize=15,horizontalalignment='center',verticalalignment='center')
                if self.mazeAgent.mazeType == 'TMaze':
                    if abs(centre[1]-1)<0.001: color = 'C0'
                    elif centre[1]>1.001: color = 'C1'
                    elif centre[1]<0.999: color = 'C2'
                    else: color = 'C3'
                else:
                    color = 'C'+str(i)
                if centresOnly == True: 
                    alpha=1
                    c='darkgrey'
                    if onepink == True:
                        if i == 30:
                            if self.mazeAgent.mazeType == 'twoRooms':
                                centre = np.array([3,1.1])
                                c = 'C3'
                    if self.mazeAgent.mazeType == 'twoRooms': 
                        s = 400; linewidth = 9
                    else: 
                        s = 130; linewidth = 4
                    if threetypes==True: 
                        alpha=0.7
                        if i%3 == 0:
                            centre -= [0.02,-0.03]
                            c='C0'
                        elif i%3 == 1:
                            c='C1'
                        elif i%3 == 2:
                            centre += [0.02,-0.03]
                            c='C3'

                    ax.scatter(centre[0],centre[1],marker='x',s=s,color=c,linewidth=linewidth,edgecolors='darkgrey',alpha=alpha)
                else:
                    circle = matplotlib.patches.Ellipse((centre[0],centre[1]), 2*self.mazeAgent.sigmas[i], 2*self.mazeAgent.sigmas[i], alpha=0.5, facecolor=color)
                    ax.add_patch(circle)
                
        saveFigure(fig, "basis")
        return fig, ax 
    
    def plotHeatMap(self,smoothing=1):
        posdata = np.stack(self.mazeAgent.history['pos'])
        bins = [int(n/smoothing) for n in list(self.mazeAgent.discreteCoords.shape[:2])]
        bins.reverse()
        hist = np.histogram2d(posdata[:,0],posdata[:,1],bins=bins)[0]
        fig, ax = self.plotMazeStructure(hist_id=-1)
        ax.imshow(hist.T, extent=self.mazeAgent.extent)
        return fig, ax



    def animateField(self, number=0,field='place',interval=100):
        if field == 'place':
            fig, ax = self.plotPlaceField(hist_id=0,number=number,show=False,save=False)
            anim = FuncAnimation(fig, self.plotPlaceField, fargs=(None, fig, ax, number, False, True, True, False), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'grid':
            fig, ax = self.plotGridField(hist_id=0,number=number,show=False,save=False)
            anim = FuncAnimation(fig, self.plotGridField, fargs=(None, fig, ax, number, False, True, True, False), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'M':
            fig, ax = self.plotM(hist_id=0,show=False,save=False,colorbar=False)
            anim = FuncAnimation(fig, self.plotM, fargs=(None, fig, ax, False,"", "Synaptic Weight Matrix \n (STDP Hebbian learning)", False,False,False,True), frames=len(self.snapshots), repeat=False, interval=interval)
        
        today = datetime.strftime(datetime.now(),'%y%m%d')
        now = datetime.strftime(datetime.now(),'%H%M')
        saveFigure(anim,saveTitle=field+"Animation",anim=True)
        return anim


    
    def plotMAveraged(self,time=None):
        # only works/defined for open loop maze
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
            snapshot = self.snapshots.iloc[hist_id]
        else:
            snapshot = self.snapshots.iloc[-1]

        M = snapshot['M'].copy()
        W = snapshot['W'].copy() 
        W_notheta = snapshot['W_notheta'].copy()
        roll = int(self.mazeAgent.nCells/2)
        M_copy, W_copy, W_notheta_copy = M.copy(), W.copy(), W_notheta.copy()
        for i in range(self.mazeAgent.nCells):
            M_copy[i,:] = np.roll(M[i,:],-i+roll)
            W_copy[i,:] = np.roll(W[i,:],-i+roll)
            W_notheta_copy[i,:] = np.roll(W_notheta[i,:],-i+roll)

        M_av,M_std = np.mean(M_copy,axis=0),np.std(M_copy,axis=0)
        W_av,W_std = np.mean(W_copy,axis=0),np.std(W_copy,axis=0)
        W_notheta_av,W_notheta_std = np.mean(W_notheta_copy,axis=0),np.std(W_notheta_copy,axis=0)
       
        W_norm = np.maximum(np.max(W_notheta_av),np.max(W_av))
        M_av,M_std = M_av/(np.max(M_av)), M_std/(np.max(M_av))
        W_av,W_std = W_av/W_norm, W_std/W_norm
        W_notheta_av,W_notheta_std = W_notheta_av/W_norm, W_notheta_std/W_norm
        x = self.mazeAgent.centres[:,0]
        x = x-x[roll]

        for i in range(len(x)):
            if x[i] > self.mazeAgent.extent[1]/2:
                x[i] = x[i] - self.mazeAgent.extent[1]

        roll = int(self.mazeAgent.nCells/2)


        fig, ax = plt.subplots(2,1,figsize=(2,2))


        Rs_wav = Rsquared(W,M)
        Rsq_wnothetaav = Rsquared(W_notheta,M)

        ax[1].plot(x,M_av,c='C0',linewidth=2, label = r" ")
        ax[0].plot(x,W_av,c='C1',label=r" ",linewidth=2)
        # ax[1].plot(x,M_theta_av,c='C0',linewidth=1.5,alpha=0.5,linestyle='dotted')
        ax[0].plot(x,W_notheta_av,c='C1',label=r" ",linewidth=1.5,alpha=0.7,linestyle='--', dashes=(1, 1))

        ax[1].fill_between(x,M_av+M_std,M_av-M_std,facecolor='C0',alpha=0.2)
        ax[0].fill_between(x,W_av+W_std,W_av-W_std,facecolor='C1',alpha=0.2)
        ax[0].fill_between(x,W_notheta_av+W_notheta_std,W_notheta_av-W_notheta_std,facecolor='C1',alpha=0.2)
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[0].set_xlim(min(x),max(x))
        ax[1].set_xlim(min(x),max(x))
        ax[0].set_ylim(min(W_av-W_std),max(W_av+W_std))
        ax[0].set_xticks([-2,-1,0,1,2])
        ax[0].set_xticklabels(['','','','',''])
        ax[1].set_xticks([-2,-1,0,1,2])
        ax[1].set_xticklabels(['','','','',''])
        ax[0].tick_params(width=2,color='darkgrey')
        ax[1].tick_params(width=2,color='darkgrey')
        plt.grid(False)

        ax[0].spines['left'].set_position('zero')
        ax[0].spines['left'].set_color('darkgrey')
        ax[0].spines['left'].set_linewidth(2)
        ax[0].spines['right'].set_color('none')        
        ax[0].spines['bottom'].set_position('zero')
        ax[0].spines['bottom'].set_color('darkgrey')
        ax[0].spines['bottom'].set_linewidth(2)
        ax[0].spines['top'].set_color('none')

        ax[1].spines['left'].set_position('zero')
        ax[1].spines['left'].set_color('darkgrey')
        ax[1].spines['left'].set_linewidth(2)
        ax[1].spines['right'].set_color('none')        
        ax[1].spines['bottom'].set_position('zero')
        ax[1].spines['bottom'].set_color('darkgrey')
        ax[1].spines['bottom'].set_linewidth(2)
        ax[1].spines['top'].set_color('none')

        ax[0].legend(frameon=False)    
        ax[1].legend(frameon=False)    
        return fig, ax

    def plotMetrics(self):
        t         = []

        W_snr         = [] 
        W_notheta_snr = []

        W_r2 = []
        W_notheta_r2  = []

        x = self.mazeAgent.centres[:,0]

        M = rowAlignMatrix(self.mazeAgent.snapshots.iloc[-1]['M'])

        for i in range(len(self.mazeAgent.snapshots)-1):
            snapshot = self.mazeAgent.snapshots.iloc[i]
            time = snapshot['t']
            if time >= 31:
                
                R2_W, R2_Wnotheta, SNR_W, SNR_Wnotheta, skew_W, skew_Wnotheta, skew_M, peak_W, peak_Wnotheta, peak_M = self.mazeAgent.getMetrics(time=time)
        
                t.append(time/60)

                W_snr.append(SNR_W)
                W_notheta_snr.append(SNR_Wnotheta)
            
                W_r2.append(R2_W)
                W_notheta_r2.append(R2_Wnotheta)

        
        snapshot = self.mazeAgent.snapshots.iloc[-1]
        time = snapshot['t']

        fig, ax = plt.subplots(2,1,figsize=(1,2),sharex=True)

        ax[1].plot(t,W_snr,c='C1',linewidth=2, label=r"$\theta$")
        ax[1].plot(t,W_notheta_snr,c='C1',linewidth=1.5,linestyle='--', dashes=(1, 1),alpha=0.7,label=r"No $\theta$")

        ax[0].plot(t,W_r2,c='C1',linewidth=2)
        ax[0].plot(t,W_notheta_r2,c='C1',linewidth=1.5,linestyle='--', dashes=(1, 1),alpha=0.7)




        ax[1].set_ylim(bottom=0,top=max(W_snr)+0.15)
        ax[0].set_ylim(bottom=0,top=1)

        ax[1].set_xticks([0,15,30])
        ax[1].set_xticklabels(["","",""])
        ax[0].set_xticks([0,15,30])
        ax[0].set_xticklabels(["","",""])

        ax[1].tick_params(width=2,color='darkgrey')
        ax[0].tick_params(width=2,color='darkgrey')
        ax[1].set_yticks([0,3,6])
        ax[1].set_yticklabels(["","",""])
        ax[0].set_yticks([0,0.5,1])
        ax[0].set_yticklabels(["","",""])

        for i in range(2):

            ax[i].spines['left'].set_position('zero')
            ax[i].spines['left'].set_color('darkgrey')
            ax[i].spines['left'].set_linewidth(2)
            ax[i].spines['right'].set_color('none')        
            ax[i].spines['bottom'].set_position('zero')
            ax[i].spines['bottom'].set_color('darkgrey')
            ax[i].spines['bottom'].set_linewidth(2)
            ax[i].spines['top'].set_color('none')
        
        return fig, ax 


    def plotFieldSilhouette(self, N=25, plot_pf=True, plot_pf_M=True, plot_rf=False):
        x = self.mazeAgent.discreteCoords[10,:,0]
        rf = self.mazeAgent.discreteStates[10,:,N]
        pf = self.mazeAgent.getPlaceFields(M=self.mazeAgent.W, threshold=0)[N][10,:]
        pf_M = self.mazeAgent.getPlaceFields(M=self.mazeAgent.M, threshold=0)[N][10,:]
        rf, pf, pf_M = rf/np.trapz(rf,x), pf/np.trapz(pf,x), pf_M/np.trapz(pf_M,x)

        fig, ax = plt.subplots(figsize=(2,0.5))
        ax.set_xlim(0,5)
        if plot_rf == True:
            ax.fill_between(x[rf>=0],rf[rf>=0],0,facecolor="C2",alpha=0.5)
        if plot_pf_M == True:
            ax.fill_between(x[pf_M>=0],pf_M[pf_M>=0],0,facecolor="C0",alpha=0.5)        
        if plot_pf == True:
            ax.fill_between(x[pf>=0],pf[pf>=0],0,facecolor="C1",alpha=0.5)
        
        hideAxes(ax)
        ax.spines['left'].set_color('none')
        ax.set_xticks([0,2.5,5])
        ax.set_yticks([])
        ax.set_xticklabels(["","",""])
        plt.tight_layout()

        return fig, ax


def fitEllipse(image, threshold=0.5,coords=None,verbose=True):
    """Takes an array (image) and fits an ellipse to it. 
    It does this by finding contours then regressing these points against the formula Ax2 + Bxy + Cy2 + Dx + Ey = 1

    Args:
        image (np.array()): The image or array to which the 
        threshold (float, optional): The relativethreshold upon whch the edges of the image will be defined. Defaults to 0.75.
        coords (np.array(image.shape,2)): The underlying coordinates of the image . Defaults to None in which case tries to get them .

    Returns:
        tuple of arrays: the x, y and z coords of the ellipse function (can be contour plotted on top of image)
    """ 

    fig2, ax2 = plt.subplots()
    cs = ax2.contour(coords[...,0],coords[...,1],image,np.array([threshold*max(image.flatten())])).collections[0].get_paths()[0].vertices
    plt.close()
    x,y = cs[:,0], cs[:,1]
    X = np.stack((x**2,x*y,y**2,x,y)).T
    F = 1
    coords_ = coords.reshape(-1,2)
    xmin,xmax,ymin,ymax=min(coords_[:,0]),max(coords_[:,0]),min(coords_[:,1]),max(coords_[:,1])
    Y = F*np.ones(X.shape[0])
    (A,B,C,D,E) = np.matmul(np.linalg.inv(np.matmul(X.T,X) + 0.0*np.identity(X.T.shape[0])),np.matmul(X.T,Y)) #least squares fit ellipse Ax2 + Bxy + Cy2 + Dx + Ey = F
    x_coord = np.linspace(xmin,xmax,1000)
    y_coord = np.linspace(ymin,ymax,1000)   
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = A * X_coord ** 2 + B * X_coord * Y_coord + C * Y_coord**2 + D * X_coord + E * Y_coord 
    #finally get eccentricity 
    m = np.array([[A,   B/2,  D/2],
                    [B/2, C,    E/2],
                    [D/2, E/2,  F]])
    if np.linalg.det(m) < 0:
        eta = 1
    else:
        eta = -1
    eccen = np.sqrt((2*np.sqrt((A-C)**2 + B**2))/(eta*(A+C)+np.sqrt((A-C)**2 + B**2)))
    if verbose == True:
        print("Eccentricity = %.3f" %eccen)

    return (X_coord, Y_coord, Z_coord), eccen

def rowAlignMatrix(M):
    M_copy = M.copy()
    roll = int(M.shape[0]/2)
    for i in range(M.shape[0]):
        M_copy[i,:] = np.roll(M[i,:],-i+roll)
    return M_copy

def saveFigure(fig,saveTitle="",transparent=True,anim=False,specialLocation=None,figureDirectory="../figures/"):
    """saves figure to file, by data (folder) and time (name) 
    Args:
        fig (matplotlib fig object): the figure to be saved
        saveTitle (str, optional): name to be saved as. Current time will be appended to this Defaults to "".
    """	

    today =  datetime.strftime(datetime.now(),'%y%m%d')
    if not os.path.isdir(figureDirectory + f"{today}/"):
        os.mkdir(figureDirectory + f"{today}/")
    figdir = figureDirectory + f"{today}/"
    now = datetime.strftime(datetime.now(),'%H%M')
    path_ = f"{figdir}{saveTitle}_{now}"
    path = path_
    i=1
    while True:
        if os.path.isfile(path+".pdf") or os.path.isfile(path+".mp4"):
            path = path_+"_"+str(i)
            i+=1
        else: break
    if anim == True:
        fig.save(path + ".mp4")
    else:
        fig.savefig(path+".pdf", dpi=400,transparent=transparent)
    
    if specialLocation is not None: 
        fig.savefig(specialLocation, dpi=400,transparent=transparent,bbox_inches='tight')

    return path

def pickleAndSave(class_,name,saveDir='../savedObjects/'):
	"""pickles and saves a class
	this is not an efficient way to save the data, but it is easy 
	this will overwrite previous saves without warning
	Args:
		class_ (any class): the class/model to save
		name (str): the name to save it under 
		saveDir (str, optional): Directory to save into. Defaults to './savedItems/'.
	"""	
	with open(saveDir + name+'.pkl', 'wb') as output:
		dill.dump(class_, output)
	return 

def loadAndDepickle(name, saveDir='../savedObjects/'):
	"""Loads and depickles a class saved using pickleAndSave
	Args:
		name (str): name it was saved as
		saveDir (str, optional): Directory it was saved in. Defaults to './savedItems/'.
	Returns:
		class: the class/model
	"""	
	with open(saveDir + name+'.pkl', 'rb') as input:
		item = dill.load(input)
	return item

def Rsquared(y1, y2):
    """R squared between two arrays 

    Args:
        y1 (np.array()): array 1
        y2 (np.array()): array 2

    Returns:
        float: R squared value between them 
    """    
    return ((1/y1.size) * np.sum((y1-np.mean(y1)) * (y2-np.mean(y2))) / (np.std(y1) * np.std(y2)))**2


def getCOM(array):
    print(array.shape)
    i_av, j_av = 0, 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            i_av += array[i,j]*i
            j_av += array[i,j]*j
    i_av /= np.sum(array)
    j_av /= np.sum(array)
    i_av = int(i_av)
    j_av = int(j_av)
    return (i_av,j_av)

def getMoment(x,y,moment=1,c=0):
    """Get moments not from sample of points but from a function (list of x, and f(x)=y)

    Args:
        x (np.array): independent variable
        y (np.array): dependent variable
        moment (int, optional): which moment ot get. Defaults to 1.
        c (float): about which point to find moment, defaults to zero
    """    
    s_x = 0
    s_y = 0
    for i in range(len(x)):
        s_x += ( (x[i] - c)**moment ) * y[i]
        s_y += y[i]
    return s_x / s_y

def getCircularMoment(theta,y,moment=1,c=0):
    Cp = np.sum(np.cos(moment*theta)*y) / np.sum(y)
    Sp = np.sum(np.sin(moment*theta)*y) / np.sum(y)
    Rp = np.sqrt(Cp**2 + Sp**2)
    if Cp > 0 and Sp > 0: 
        Tp = np.arctan(Sp/Cp)
    elif Cp < 0: 
        Tp = np.arctan(Sp/Cp) + np.pi
    elif Sp < 0 and Cp > 0: 
        Tp = np.arctan(Sp/Cp) + 2*np.pi
    return Rp, Tp



def getSkewness(y,circular=False):
    if not np.all(y>=0):
        y = np.maximum(y,0)
    x = np.linspace(0,2*np.pi,len(y))
    if circular == False: 
        mean = getMoment(x,y)
        std = np.sqrt(getMoment(x,y,moment=2,c=mean))
        skewness = getMoment(x,y,moment=3,c=mean) / std**3
    if circular == True: #NCSS Statistical Software NCSS.com, Chapter 230, Circular Data Analysis, https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Circular_Data_Analysis.pdf
        R1, T1 = getCircularMoment(x,y,moment=1)
        R2, T2 = getCircularMoment(x,y,moment=2)
        V = 1-R1
        skewness = R2*np.sin(2*T1-T2) / (1-R1)**(3/2)

    return skewness


def getPeak(x,y,smooth=True):
    if smooth == True: 
        y_smooth = np.empty_like(y)
        for i in range(len(y)):
            y_smooth[i] = np.mean(y[max(0,i-1):min(i+2,len(y))])
        peak = x[np.argmax(y_smooth)]
    else:
        peak = x[np.argmax(y)]
    return peak 
