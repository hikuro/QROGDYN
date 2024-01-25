########################### (quasi)proggenHP.py ###############################
#
#    A program to generate geoPlusVel.
#    This code is based on proggenHP ver 2014, written by Daniel A. Singleton
#
#    written by Hiroaki Kurouchi
#    ver 06/12/2018 Standard process was implemented
#    ver 06/19/2018 Classical 2 was implemented, organized into function 
#    ver 08/08/2018 Fitgeometry is implemented 
#    ver 11/26/2018 Modified for SPTS sampling of Yang, Doubleday and Houk
#                   See JCTC, 2015, 5606 for details.
#    ver 12/05/2018 Bug was fixed
#    ver 02/22/2019 Modified for GRRM 
#
################################################################################

import random
import os
import progread
import gausslog
import copy
import numpy as np
import progdynstarterHP
import sys
import solvgen

#Define Constants
conver1=4.184E26 #dividing by this converts amu angs^2 /s^2 to kcal/mol
c=29979245800; h=6.626075E-34; avNum=6.0221415E23
RgasK=0.00198588; RgasJ=8.31447

# This function reads freqinHP and generates geoPlusVel.
# In order to include solvent information, prepare solvgeoPlusVel and
# use solventread = 1
def proggenHP(programdir,origdir,freqfile):
    #initialize parameters
    initialDis   = int(progread.confread_each(origdir,"initialdis",0)[1])
    diag         = int(progread.confread_each(origdir,"diagnostics",0)[1])
    timestep     = float(progread.confread_each(origdir,"timestep",1E-15)[1])
    scaling      = float(progread.confread_each(origdir,"scaling",1.0)[1])
    temp         = float(progread.confread_each(origdir,"temperature",298.15)[1])
    searchdir    = progread.confread_each(origdir,"searchdir","positive")[1]
    classical    = int(progread.confread_each(origdir,"classical",0)[1])
    numimag      = int(progread.confread_each(origdir,"numimag",1)[1])
    highlevel    = progread.confread_each(origdir,"highlevel",999)[1]
    boxon        = int(progread.confread_each(origdir,"boxon",0)[1])
    boxsize      = float(progread.confread_each(origdir,"boxsize",10)[1])
    DRP          = int(progread.confread_each(origdir,"DRP",0)[1])
    singletraj   = int(progread.confread_each(origdir,"singletraj",0)[1])
    if DRP == 1 or singletraj == 1: 
        classical = 1
    maxAtomMove  = float(progread.confread_each(origdir,"maxatommove",0.1)[1])
    cannonball   = progread.confread_each(origdir,"cannonball",0)[1]
    geometry     = progread.confread_each(origdir,"geometry","nonlinear")[1]
    disMode      = progread.confread_each_list(origdir,"displacements")
    controlPhase = progread.confread_each_list(origdir,"controlphase")
    rotationmode = progread.confread_each(origdir,"rotationmode",0)[1]
    ONIOM        = int(progread.confread_each(origdir,"ONIOM",0)[1])
    solventread  = int(progread.confread_each(origdir,"solventread",0)[1])
    spts         = int(progread.confread_each(origdir,"spts",0)[1])
    highatomnum  = int(progread.confread_each(origdir,"highatomnum",0)[1])
    grrminput    = int(progread.confread_each(origdir,"grrminput",0)[1])

    #Define Constants
    classicalSpacing = 2

    if spts == 1:
        print("spts calculation is applied")
        freqfile = origdir + "spts.out"
    print("freqfile="+str(freqfile))
    # put atomnum, atomweight, geometries into ndarray, also figure out number of atoms

    if grrminput == 0:
        geoArr,atSym,atWeight = gausslog.structurereader(freqfile)
        mode,freq,force,redMass = gausslog.normal_mode_reader(freqfile)
    elif grrminput == 1:
        geoArr,atSym,atWeight,mode,freq,force,redMass = gausslog.grrm_freqcalcreader(freqfile)

    print(freq)
    if diag == 1:
        with open (origdir+"diagnostics",mode='a') as dw:
            dw.write("freq ")
            for i in range(len(freq)):
                dw.write(str(freq[i])+" ")
            dw.write("\n")
            dw.flush()


    # Because degree of freedum in ONIOM frequency calculation is 3N (including negative freq),
    # The dimention of mode, freq, force, and redMass should be reduced
    if ONIOM == 1 and highatomnum > 0:
        mode    = mode[:3*highatomnum]
        freq    = freq[:3*highatomnum]
        force   = force[:3*highatomnum]
        redMass = redMass[:3*highatomnum]
    elif ONIOM == 1 and highatomnum == 0:
        print("turn off ONIOM or input highatomnum")
        sys.exit()

    # Replace negative frequencies by 2.0
    freq = (freq < 0) * 2 + (freq > 0) * freq
    # Initialize random array
    randArr,randArrB,randArrC,randArrD = makerandArr(freq)
    if diag == 1:
        with open (origdir+"diagnostics",mode='a') as dw:
            dw.write("randArr  randArrB  randArrC  randArrD ")
            for i in range(len(freq)):
                dw.write(str(randArr[i])+" "+str(randArrB[i])+" "+str(randArrC[i])+" "+str(randArrD[i])+"\n")
            dw.write("\n")
            dw.flush()

    ### Calculate excitation of modes
    zpeJ =  0.5 * h * c * freq # units, J/molecule
    geoArrOrig = copy.deepcopy(geoArr)
    # gain force = 0 a littile it so as to avoid divNull.
    force = (force == 0.0) * 0.00001 + force
    # erace negative energy
    zpeJ = (zpeJ > 0) * zpeJ
    if classical == 1:
        zpeK = np.array([ 0.5*h*c*classicalSpacing for i in range(len(freq))])
    zpeK = zpeJ * avNum/4184 # units kcal/mol
    vibN = np.zeros([len(freq)])
    randArr = np.random.rand(len(freq))
    if float(temp) >= 10.0:
        zpeRat = np.exp((-2*zpeK)/(RgasK*float(temp)))
        zpeRat -= (zpeRat == 1.0) * 0.00000000001
        Q = 1/(1-zpeRat)
        tester = 1/Q
        # get up to 4000 excitations of low modes
        for i in range(len(vibN)):
            j = 1
            while j <= 4000 * zpeRat[i] + 2:
                if randArr[i] > tester[i]:
                    vibN[i] += 1
                tester[i] += zpeRat[i]**j/Q[i]
                j += 1
    # Calculate total energy desired for molecule
    modeEn  = (zpeJ * 1E18) * (2 * vibN + 1) # unit: mDyne
    modeEnK = zpeK * (2 * vibN +1)
    if classical == 1:
        modeEn  = (zpeJ * 1E18) * 2 * vibN
        modeEnK = zpeK * 2 * vibN
    desiredModeEnK = np.sum(modeEnK)
    # treat mode with freq <10 as translations, igonore the ZPE
    modeEn = (freq < 10) * (zpeJ * 1E18) * 2 * vibN + (freq >= 10) * modeEn
    if singletraj == 1:
        modeEn[1:] = 0 
    maxShift = (2 * modeEn / force) **.5
    shift = np.zeros([len(freq)]) +(freq >= 10) *\
            ((initialDis == 3) * (maxShift * np.sin(2 * np.pi * randArrC))
            +(initialDis == 2) * maxShift * randArrD
            +(initialDis == 1) * maxShift * (2 * (randArrC - 0.5)))

    #print("shift",shift)    
    #print("mode",mode)
    # We will ignore disMode because it is just a special option
    # if it is in need, write it in python style e.g. slicing parameters
    # Then, we will add the shifts up
#    shiftmode,geoArr = 0,0
    shiftMode = 0
    if classical != 2:
        shiftMode = (shift * mode.T).T
        geoArr += np.sum(shiftMode,axis=0)

    # Start toward velocities
    kinEn = 1E5 * (modeEn - 0.5 * force * (shift**2))
    vel   = (2 * kinEn/(redMass / avNum))**.5  # in angstrom/s

    # use searchdir in configurationfile to control the direction of the trajectories
    if numimag > 0:
        randArrB[0] = 1
    vel = (2 * (randArrB > 0.5) -1) * vel 
    if numimag > 0 and searchdir == "negative":
        vel[0] = -vel[0]

    # Controlphase section
    for i in range(len(freq)):
        if controlPhase[i+1] == "positive":
            vel[i] = np.abs(vel[i])
        elif controlPhase[i+1] == "negative":
            vel[i] = -np.abs(vel[i])

    # multiply each of the modes by its velocity and add them up
    velMode,velArr = 0,0
    if int(classical) != 2:
        velMode = timestep * (vel * mode.T).T 
        velArr  = np.sum(velMode,axis=0)
    # Now we start to classical == 2 for solvent system
    elif int(classical) == 2:
        # Originally,
        # degFreedomEnK=temp*RgasK
        # degFreedomEnJ=degFreedomEnK/(avNum/4184)
        # cartEn=degFreedomEnJ*1E18
        # kinEnCart=100000*cartEn
        kinEnCart = 1E23 * temp * RgasK / (avNum / 4184)
        velArr = (np.random.rand(len(geoArr),len(geoArr[0])) > 0.5) * 2 -1
        velArr = timestep * (2*kinEnCart*avNum)**.5 * (velArr.T/atWeight).T

    # calculate KE
    KEinitmodes = np.sum(0.5*atWeight*(np.sum(velArr**2,axis=1))/((timestep**2)*conver1))
    #print("KEinitmodes",KEinitmodes)
    # Add molecular rotation if requested
    # This algorithm is applicable only when axes of the initial coordinate is principal 
    # axes of moment of inertia. This is also true for the original PROGDYN.
    randArrR = np.random.rand(6)
    rotEdesired = 0
    if int(rotationmode) > 0:
        velArr,rotEdesired = addRotation(geoArrOrig,atWeight,timestep,temp,randArrR,velArr)
    # The cannonball will be implemented someday.

    # Calculate total KE = KEinitmodes + rotE
    KEinittotal = np.sum(0.5*atWeight*(np.sum(velArr**2,axis=1))/(timestep**2*conver1))
#    print("KEinittotal",KEinittotal)    
    # Here we read ZPE from freqinHP
    zpeGauss, zpePlusE = gausslog.zpereader(freqfile)
    potentialE = zpePlusE - zpeGauss

    # Read solvent information for ONIOM calculation
    # In case that the geometry in freqinHP and solvgeoPlusVel are different:
    # (1) The geoArr and velArr will be moved and rotated to fit to solvgeoPlusVel when spts = 0.
    # (2) The solventgeoPlusVel will be moved and rotated to fit to spts.out when spts = 1.

    # Before getting solvent velocity information, let's make solventgeoPlusVel
    if ONIOM == 1 and solventread == 1 and os.path.exists(origdir+"solvtraj") == True\
        and os.path.exists(origdir+"solventgeoPlusVel") == False:
        solvgen.solvgen(origdir,"solvtraj")

    if solventread == 1 and spts == 0:
        s_geoArr,s_velArr,atSym,atWeight = progread.solventgeoPlusVelread(origdir)
        geoArr,velArr = fitgeometry(atWeight[:len(geoArr)],geoArr,velArr,\
                        s_geoArr[:len(geoArr)],s_velArr[:len(geoArr)])
        newgeoArr,newvelArr = np.zeros([len(s_geoArr),3]) ,np.zeros([len(s_velArr),3])  
        newgeoArr[:len(geoArr)] = geoArr
        newvelArr[:len(velArr)] = velArr
        newgeoArr[len(geoArr):] = s_geoArr[len(geoArr):]
        newvelArr[len(velArr):] = s_velArr[len(velArr):]
        geoArr = copy.deepcopy(newgeoArr)
        velArr = copy.deepcopy(newvelArr)

    if solventread == 1 and spts == 1:
        s_geoArr,s_velArr,atSym,atWeight = progread.solventgeoPlusVelread(origdir)
        geoArr_solv,velArr_solv =  \
                            fitgeometry(atWeight[highatomnum:],s_geoArr[highatomnum:],
                            s_velArr[highatomnum:],geoArr[highatomnum:],velArr[highatomnum:])
        velArr[highatomnum:] = velArr_solv
        print("velArr",velArr)


    geoPlusVelwrite(origdir,geoArr,atSym,atWeight,velArr,vibN,vel,shift,disMode,freq,initialDis,\
                    randArr,randArrB,randArrC,randArrD,temp,classical,timestep,numimag,desiredModeEnK,\
                    KEinitmodes,KEinittotal,rotEdesired,cannonball,boxon,boxsize,DRP,maxAtomMove,\
                    potentialE,zpeGauss,zpePlusE)

    return geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE

def makerandArr(freq):
    print(freq)
    randArr  = np.random.rand(len(freq))
    randArrB = np.random.rand(len(freq))
    randArrC = np.random.rand(len(freq))
    # randArrD is special random number to make QM-like distribution
    # original algorithm was a little bit modified
    randArrD = np.zeros(len(freq))
    for i in range(len(freq)):
        while True:
            tempNum = 2*(np.random.rand() - 0.5)
            prob = np.exp(-(tempNum**2))
            randnum = np.random.rand()
            if randnum < prob:
                randArrD[i] = tempNum
                break
    print(randArr)

    return randArr,randArrB,randArrC,randArrD

def fitgeometry(atWeight,geoArr,velArr,s_geoArr,s_velArr):
    # calculate center of mass(COM)
    print(geoArr)
    print(s_geoArr)

    geoArr_COM = np.sum((atWeight * geoArr.T).T,axis=0)/np.sum(atWeight)
    s_geoArr_COM = np.sum((atWeight * s_geoArr.T).T,axis=0)/np.sum(atWeight)
    # set COM to the original point
    geoArr = geoArr - geoArr_COM
    s_geoArr = s_geoArr - s_geoArr_COM
    # calculate rotation matrix
    rotmat = np.identity(3)
    # Optimize rotation matrix by simulated annealing
    # But without transition to unstable state
    cycle = 1 
    cool = 0.9998
    T = 10
    Dev = np.std(geoArr - s_geoArr)
    while Dev > 0.000001:
        Rot = Func_Rotation((np.random.rand(3) -0.5) * T / cycle)
        newRot = np.dot(Rot,rotmat)
        newDev = np.std(np.dot(newRot,geoArr.T).T - s_geoArr)
        if newDev < Dev:
            Dev = newDev
            rotmat = newRot
            print(cycle," ", Dev)
        T *= cool
        cycle += 1
        if cycle > 50000:
            print("optimization of  freqinHP geometry reached limit of cycle number")
            print("Minimum Deviation",Dev)
            if Dev > 0.01:
                print("input structure might be wrong or you should switch displacement off")
                sys.exit()
            break
    #print("Dev",Dev)
    #print("cycle",cycle)
    #print(np.dot(rotmat,geoArr.T).T)
    #print(s_geoArr)
    geoArr = np.dot(rotmat,geoArr.T).T + s_geoArr_COM
    velArr = np.dot(rotmat,velArr.T).T
    #print(geoArr)
    return geoArr,velArr 


def Func_Rotation(Var):
    RotX = np.array([[1,0,0],[0,np.cos(Var[0]),-np.sin(Var[0])],[0,np.sin(Var[0]),np.cos(Var[0])]])
    RotY = np.array([[np.cos(Var[1]),0,np.sin(Var[1])],[0,1,0],[-np.sin(Var[1]),0,np.cos(Var[1])]])
    RotZ = np.array([[np.cos(Var[2]),-np.sin(Var[2]),0],[np.sin(Var[2]),np.cos(Var[2]),0],[0,0,1]])
    Rot = np.dot(np.dot(RotX,RotY),RotZ)
    return Rot


def addRotation(geoArrOrig,atWeight,timestep,temp,randArrR,velArr):
    rotateX,rotateY,rotateZ = np.zeros([len(geoArrOrig),3]),\
                              np.zeros([len(geoArrOrig),3]),np.zeros([len(geoArrOrig),3])
    for i in range(len(geoArrOrig)):
        rotateX[i,1] = -geoArrOrig[i,2]
        rotateX[i,2] =  geoArrOrig[i,1]
        rotateY[i,0] = -geoArrOrig[i,2]
        rotateY[i,2] =  geoArrOrig[i,0]
        rotateZ[i,0] = -geoArrOrig[i,1]
        rotateZ[i,1] =  geoArrOrig[i,0]
    eRotX = np.sum(0.5*atWeight*(rotateX**2).T/(timestep**2*conver1))
    eRotY = np.sum(0.5*atWeight*(rotateY**2).T/(timestep**2*conver1))
    eRotZ = np.sum(0.5*atWeight*(rotateZ**2).T/(timestep**2*conver1))

    # decide how much energies we want in each rotation
    keRx = (eRotX >= 1) * -0.5*RgasK*temp*np.log(1-randArrR[0])
    keRy = (eRotY >= 1) * -0.5*RgasK*temp*np.log(1-randArrR[1])
    keRz = (eRotZ >= 1) * -0.5*RgasK*temp*np.log(1-randArrR[2])
    rotEdesired = keRx + keRy + keRz
    signX = np.sign(randArrR[3]-1)
    signY = np.sign(randArrR[4]-1)
    signZ = np.sign(randArrR[5]-1)
    keRx += (keRx < 1) * 1
    keRy += (keRy < 1) * 1
    keRz += (keRz < 1) * 1
    scaleX = (keRx/eRotX)**.5
    scaleY = (keRy/eRotY)**.5
    scaleZ = (keRz/eRotZ)**.5
    rotateX *= scaleX * signX
    rotateY *= scaleY * signY
    rotateZ *= scaleZ * signZ

    # Here sum up all the rotation modes
    velArr += rotateX + rotateY + rotateZ

    return velArr,rotEdesired

def geoPlusVelwrite(origdir,geoArr,atSym,atWeight,velArr,vibN,vel,shift,disMode,freq,initialDis,\
                    randArr,randArrB,randArrC,randArrD,temp,classical,timestep,numimag,desiredModeEnK,\
                    KEinitmodes,KEinittotal,rotEdesired,cannonball,boxon,boxsize,DRP,maxAtomMove,\
                    potentialE,zpeGauss,zpePlusE):
    # Here we generate geoPlusVel 
    with open(origdir+"geoPlusVel",mode='w') as gv:
        gv.write(str(len(geoArr))+"\n")  
        for i in range(len(geoArr)):
            gv.write(" {:2}".format(atSym[i])+" "+"{: .7f} {: .7f} {: .7f} {: .5f}".format(\
                     geoArr[i][0],geoArr[i][1],geoArr[i][2],atWeight[i])+"\n")
        for i in range(len(geoArr)):
            gv.write("{: .8f} {: .8f} {: .8f}".format(velArr[i][0],velArr[i][1],velArr[i][2])+"\n")
        if classical != 2:
            for i in range(len(freq)):
                if initialDis == 0:
                    gv.write("{: .6f} {: .6f}   {:>4}   {: .4e} {: .6f}  {: >2}".format(\
                    randArr[i],randArrB[i],int(vibN[i]),vel[i],shift[i],disMode[i]) + "\n")
                if initialDis == 1:
                    gv.write("{: .6f} {: .6f}   {:>4}   {: .4e} {: .6f}  {: >2}".format(\
                    randArr[i],randArrC[i],int(vibN[i]),vel[i],shift[i],disMode[i]) + "\n")
                if initialDis == 2:
                    gv.write("{: .6f} {: .6f}   {:>4}   {: .4e} {: .6f}  {: >2}".format(\
                    randArr[i],randArrD[i],int(vibN[i]),vel[i],shift[i],disMode[i]) + "\n")
                if initialDis == 3:
                    gv.write("{: .6f} {: .6f}   {:>4}   {: .4e} {: .6f}  {: >2} {: .6f}".format(\
                    randArr[i],randArrC[i],int(vibN[i]),vel[i],shift[i],disMode[i],
                    np.sin(randArrC[i]*2*np.pi)) + "\n")

        gv.write("temp "+str(temp)+"\ninitialDis "+str(initialDis)+"\nclassical "+str(classical)+
                 "\ntimestep "+str(timestep)+"\nnumimag "+str(numimag)+"\nTotal mode energy desired= "+
                 "{: .3f}".format(desiredModeEnK)+"\nKE initial from modes= "+"{: .3f}".format(KEinitmodes)+
                 "  KE initial total= "+"{: .3f}".format(KEinittotal)+"  Rotational Energy desired= "+
                 "{: .3f}".format(rotEdesired) + "\n")
        if cannonball > 0:
            gv.write("cannonball "+str(cannonball)+
            " cannon Energy= ""{: .3f}".format(KEinittotal-KEinitmodes)+"\n")
        if boxon > 0:
            gv.write("boxsize "+str(boxsize)+"\n")
        if DRP > 1:
            gv.write("DRP " + str(DRP)+" maxAtomMove "+str(maxAtomMove)+"\n")
        gv.write("Gaussian zpe= "+"{: .6f}".format(zpeGauss)+" or "+
                 "{: .6f}".format(zpeGauss*627.509)
                 +" kcal/mol  E + zpe= "+"{: .6f}".format(zpePlusE)+" potential E= "+
                 "{: .6f}".format(zpePlusE-zpeGauss)+"\n\n")
                
        gv.flush()

if __name__ == "__main__":
    origdir = os.getcwd()
    if origdir[-1] != "/":
        origdir = origdir + "/"

    proggenHP(progdynstarterHP.programdir,origdir,progdynstarterHP.freqfile)




