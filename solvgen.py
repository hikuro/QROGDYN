################################ solvgen.py ###################################
#
#    Inputfile for solvent relaxation, geoPlusVel generator.
#    This code is based on proggenHP ver 2014, written by Daniel A. Singleton
#    Only the first layer is fixed
#
#    Type "python3 (path)/solvgen.py" to run this program.
#    This code generates "solventgeoPlusVel" which will be used by proggenHP.py
#    when "solvtraj" file exists in the current directory.
#    This code generates "geoPlusVel" when solvgeo.pdb exists in the current
#    directory and solvtraj does not exists there. The "geoPlusVel" is made 
#    with completely random motion without any displacement change.
#    
#    --How to use this program in the series of calculation--
#    (1) Prepare initial geometry by packmol program and save it as "solvgeo.pdb"
#    (2) Run this program to make "geoPlusVel"
#    (3) With the "geoPlusVel" file, start runnning quasiprogdyn to make "solvtraj"
#    (4) (It will be automatically done if you forget)
#        Run this program to make "solventgeoPlusVel"
#    (5) Now you can start trajectory calculation. Run the quasiprogdyn with 
#        option "solventread 1" in progdyn.conf
#
#    written by Hiroaki Kurouchi
#    ver  06/22/2018 standard process was implemented
#    ver  04/17/2019 centeratomPhaseMake was implemented
#    ver  04/12/2021 centeratomfix was modified to fix outer sphere
#
################################################################################

import random
import os
import progread
import gausslog
import copy
import numpy as np
import progdynstarterHP
import linecache
import sys

#Define Constants
conver1=4.184E26 #dividing by this converts amu angs^2 /s^2 to kcal/mol
c=29979245800; h=6.626075E-34; avNum=6.0221415E23
RgasK=0.00198588; RgasJ=8.31447

# Make solventgeoPlusVel file so as to use for geoPlusVel
def solvgen(origdir,filename):
    print("solvgen is called")
    geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir,"solvgeoPlusVel")
    if len(geoArr) < 1:
        print("No solvgeoPlusVel")
    else:
        print("solventgeoPlusVel is successfully read")
    timestep   = float(progread.confread_each(origdir,"timestep",1E-15)[1])
    atomnum    = int(linecache.getline(origdir+filename,1).split()[0])
    olderArr   = np.zeros([atomnum,3])
    oldArr     = np.zeros([atomnum,3])
    linenum    = sum(1 for line in open(origdir+filename))
    num_of_str = int(linenum / (atomnum + 2))
#    print(atSym,timestep,atomnum,oldArr) 
    for i in range(atomnum):
        lineinfo = linecache.getline(origdir+filename,(num_of_str-2) * (atomnum+2) + i + 3).split()
        oldArr[i][0],oldArr[i][1],oldArr[i][2] = float(lineinfo[1]),float(lineinfo[2]),float(lineinfo[3])
        lineinfo = linecache.getline(origdir+filename,(num_of_str-3) * (atomnum+2) + i + 3).split()
        olderArr[i][0],olderArr[i][1],olderArr[i][2] = float(lineinfo[1]),float(lineinfo[2]),float(lineinfo[3])

#    print("starting geometry",oldArr)
    if os.path.exists(origdir+"status"):
        os.remove(origdir+"status")
    progread.statusread(origdir,"bypassproggen","off")
    progread.statusread(origdir,"runpointnum","1")

    velArr = (oldArr - olderArr) / (timestep/1E-15)
    #os.rename(origdir+"geoPlusVel",origdir+"geoPlusVel_solventequiv")
    geoPlusVelwrite(origdir,oldArr,atSym,atWeight,velArr,filename="solventgeoPlusVel")

# Make geoPlusVel randomly using pdb file information
def solvgen_rand(origdir,geofile):
    # initialize parameters
    timestep     = 1E-15 
    # we will start from 1000K
    temp         = 1000.0 

    # put atomnum, atomweight, geometries into ndarray, also figure out number of atoms
    geoArr,molnum,layernum,atSym,atWeight = progread.pdbread(geofile)

    # Initialize random array
    randArr,randArrB,randArrC,randArrD = makerandArr(atSym)

    # Now we start to classical == 2 for solvent system
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
    # This algorythm is applicable only when axes of the initial coordinate is principal 
    # axes of moment of inertia. This is also true for the original PROGDYN.

    # Read solvent information for ONIOM calculation
    geoPlusVelwrite(origdir,geoArr,atSym,atWeight,velArr)
    if os.path.exists(origdir+"status"):
        os.remove(origdir+"status")
    progread.statusread(origdir,"bypassproggen","on")
    progread.statusread(origdir,"runpointnum","1")

    if os.path.exists(origdir+"thermoinfo"):
        os.remove(origdir+"thermoinfo")

    return geoArr,velArr,atSym,atWeight

def makerandArr(freq):
    randArr  = np.random.rand(len(freq))
    randArrB = np.random.rand(len(freq))
    randArrC = np.random.rand(len(freq))
    # randArrD is special random number to make QM-like distribution
    # original algorythm was a little bit modified
    randArrD = np.zeros(len(freq))
    for i in range(len(freq)):
        prob = np.exp(-np.random.rand()**2)
        while True:
            randnum = np.random.rand()
            if randnum < prob:
                randArrD[i] = randnum
                break
    return randArr,randArrB,randArrC,randArrD


def geoPlusVelwrite(origdir,geoArr,atSym,atWeight,velArr,filename="geoPlusVel"):
    # Here we generate geoPlusVel 
    with open(origdir+filename,mode='w') as gv:
        gv.write(str(len(geoArr))+"\n")  
        for i in range(len(geoArr)):
            gv.write(" {:2}".format(atSym[i])+" "+"{: .7f} {: .7f} {: .7f} {: .5f}".format(\
                     geoArr[i][0],geoArr[i][1],geoArr[i][2],atWeight[i])+"\n")
        for i in range(len(geoArr)):
            gv.write("{: .8f} {: .8f} {: .8f}".format(velArr[i][0],velArr[i][1],velArr[i][2])+"\n")

def centeratomPhaseMake(origdir,freqfile):
    print("\nNow Making centeratomPhase")
    highatomnum  = int(progread.confread_each(origdir,"highatomnum",0)[1])

    geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
                                   = progread.geoPlusVelread(origdir,"geoPlusVel") 

    geoArr = geoArr[:highatomnum]
    o_geoArr,atSym,atWeight = gausslog.structurereader(freqfile)
    mode,freq,force,redMass = gausslog.normal_mode_reader(freqfile)

    # rotates mode and erace the motion if the frequencies are negative
    rotated_mode = fitgeometry(atWeight,o_geoArr,geoArr,mode)
    phase = np.array([np.random.rand()*2*np.pi for i in range(len(mode))]) 
    
    # force:mDyn/A = 1E-8 N/A = 100 N/m = 100 kg/s^2. redMass:AMU = 1.6605402E-27 kg
    angular_velocity = 1E-15 * (force * 100 /(redMass * 1.6605402E-27))**0.5 #unit:fs^-1

    # h=6.626075E-34 m^2kg/s
    amplitude = 1e10 * (h/(2*np.pi*angular_velocity * 1E15 * redMass * 1.6605402E-27))**0.5  * (freq > 0)  #unit:angstrom
#    amplitude = 1e10 * (h/(angular_velocity * 1E15 * redMass * 1.6605402E-27))**0.5 / (2 * np.pi) #unit:angstrom

    with open(origdir + "centeratomPhase",mode="w") as cw:
        cw.write(str(len(amplitude))+" "+str(len(rotated_mode[0]))+"\n")
        # First, write Amplitude
        for i in range(len(amplitude)):
            cw.write("amplitude "+str(i)+" "+str(amplitude[i])+"\n")

        # Next, write angular velocity
        for i in range(len(angular_velocity)):
            cw.write("angvel "+str(i)+" "+str(angular_velocity[i])+"\n")
   
        # Then, write phase
        for i in range(len(phase)):
            cw.write("phase "+str(i)+" "+str(phase[i])+"\n")

        # Write rotated modes
        for i in range(len(rotated_mode)):
            for j in range(len(rotated_mode[0])):
                cw.write("rotated_mode "+str(i)+" "+str(j)+\
                " {: .7f} {: .7f} {: .7f}\n".format(rotated_mode[i][j][0],\
                rotated_mode[i][j][1],rotated_mode[i][j][2]))

        cw.flush()

def fitgeometry(atWeight,o_geoArr,geoArr,mode):
    # calculate center of mass(COM)
    geoArr_COM = np.sum((atWeight * geoArr.T).T,axis=0)/np.sum(atWeight)
    o_geoArr_COM = np.sum((atWeight * o_geoArr.T).T,axis=0)/np.sum(atWeight)

    # set COM to the original point
    geoArr = geoArr - geoArr_COM
    o_geoArr = o_geoArr - o_geoArr_COM

#    print(geoArr)
#    print(o_geoArr)

    # calculate rotation matrix
    rotmat = np.identity(3)
    # Optimize rotation matrix by simulated annealing
    # But without transition to unstable state
    cycle = 1
    cool = 0.9998
    T = 10
    Dev = np.std(geoArr - o_geoArr)
    while Dev > 0.000001:
        Rot = Func_Rotation((np.random.rand(3) -0.5) * T / cycle)
        newRot = np.dot(Rot,rotmat)
        newDev = np.std(np.dot(newRot,o_geoArr.T).T - geoArr)
        if newDev < Dev:
            Dev = newDev
            rotmat = newRot
#            print(cycle," ", Dev)
        T *= cool
        cycle += 1
        if cycle > 50000:
            print("optimization of  freqinHP geometry reached limit of cycle number")
            print("Minimum Deviation",Dev)
            if Dev > 0.001:
                print("input structure might be wrong or you should switch displacement off")
                sys.exit()
            else:
                print("centeratomPhaseMake is created safely")
            break

    rotated_mode = np.array([np.dot(rotmat,mode[N].T).T for N in range(len(mode))])

    return rotated_mode 

def Func_Rotation(Var):
    RotX = np.array([[1,0,0],[0,np.cos(Var[0]),-np.sin(Var[0])],[0,np.sin(Var[0]),np.cos(Var[0])]])
    RotY = np.array([[np.cos(Var[1]),0,np.sin(Var[1])],[0,1,0],[-np.sin(Var[1]),0,np.cos(Var[1])]])
    RotZ = np.array([[np.cos(Var[2]),-np.sin(Var[2]),0],[np.sin(Var[2]),np.cos(Var[2]),0],[0,0,1]])
    Rot = np.dot(np.dot(RotX,RotY),RotZ)
    return Rot


if  __name__ == "__main__":
    origdir = os.getcwd()  
    pdbfile  = progdynstarterHP.pdbfile
    if origdir[-1] != "/":
        origdir = origdir + "/"
    centeratomfix = int(progread.confread_each(origdir,"centeratomfix",0)[1])
    if os.path.exists(origdir+"solvtraj") == True:
        print("making solventgeoPlusVel")
        solvgen(origdir,"solvtraj")
    elif os.path.exists(origdir+pdbfile) == True:
        print("making geoPlusVel using pdb input file")
        solvgen_rand(origdir,origdir+pdbfile)
        if centeratomfix == 2:
            centeratomPhaseMake(origdir,progdynstarterHP.freqfile)
    else:
        print("no solvtraj nor pdbfile")





