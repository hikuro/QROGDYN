################################## progread.py #################################
#
#    A program to read files for trajectory calculation
#    Sometimes used for writing into files,
#    so the name should have been progreadandwrite.py
#
#    written by Hiroaki Kurouchi
#    ver 06/17/2018
#    ver 08/09/2018 statusread is mainly changed
#    ver 01/28/2019 statusread was modified for fail-safe calculation
#
################################################################################
import os
import sys
import linecache
import numpy as np
import progdynstarterHP
import time

#*** geoPlusVel
def geoPlusVelread(origdir,filename="geoPlusVel"):
    if os.path.exists(origdir+filename) == True:
        #print("geoPlusVelread start")
        atomNum = int(linecache.getline(origdir+filename,1).split()[0])
        geoArr,velArr = np.zeros([atomNum,3]),np.zeros([atomNum,3])
        atSym,atWeight = [0 for i in range(atomNum)],np.zeros([atomNum])
        desiredModeEnK,KEinitmodes,KEinittotal,potentialE = 0,0,0,0
        for i in range(atomNum):
            lineinfo = linecache.getline(origdir+filename,i+2).split()
            atSym[i] = lineinfo[0]
            atWeight[i] = float(lineinfo[4])
            for j in range(3):
                geoArr[i][j] = lineinfo[j+1]
        for i in range(atomNum):
            lineinfo = linecache.getline(origdir+filename,i+2+atomNum).split()
            for j in range(3):
                velArr[i][j] = lineinfo[j]
        linenum  = sum(1 for line in open(origdir+filename))
        for i in range(linenum):
            lineinfo = linecache.getline(origdir+filename,i+1).split()
            try:
                if lineinfo[3] == "desired=":
                    desiredModeEnK = float(lineinfo[4])
            except:
                pass
            try:
                if lineinfo[3] == "modes=":
                    KEinitmodes,KEinittotal = float(lineinfo[4]), float(lineinfo[8])
            except:
                pass
            try:
                if lineinfo[10] == "potential":
                    potentialE = float(lineinfo[12])
            except:
                pass
    else:
        with open(origdir+"Errormessage",mode='a') as df:
            df.write("The loop is killed because geoPluSVel does not exist\n")
            df.flush()
        sys.exit()

    linecache.clearcache()
    return geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE

#*** solventgeoPlusVel
def solventgeoPlusVelread(origdir):
    if os.path.exists(origdir+"solventgeoPlusVel") == True:
        atomNum = int(linecache.getline(origdir+"solventgeoPlusVel",1).split()[0])
        geoArr,velArr = np.zeros([atomNum,3]),np.zeros([atomNum,3])
        atSym,atWeight = [0 for i in range(atomNum)],np.zeros([atomNum])
        for i in range(atomNum):
            lineinfo = linecache.getline(origdir+"solventgeoPlusVel",i+2).split()
            atSym[i] = lineinfo[0]
            atWeight[i] = float(lineinfo[4])
            for j in range(3):
                geoArr[i][j] = lineinfo[j+1]
        for i in range(atomNum):
            lineinfo = linecache.getline(origdir+"solventgeoPlusVel",i+2+atomNum).split()
            for j in range(3):
                velArr[i][j] = lineinfo[j]
    else:
        with open(origdir+"Errormessage",mode='a') as df:
            df.write("The loop is killed because solventgeoPluSVel does not exist\n")
            df.flush()
        sys.exit()

    linecache.clearcache()
    return geoArr,velArr,atSym,atWeight

#*** progdyn.conf
def confread_each(path,key,default=0):
    param = [default for i in range(10)] 
    conffile = progdynstarterHP.conffile
    if os.path.exists(path) == True:
        key = str(key)
        filename = path + conffile
        linenum  = sum(1 for line in open(filename))
        for line in range(linenum):
            Linedata = linecache.getline(filename,line+1).split()
            try:
                if len(Linedata) == 0 or len(Linedata) == 1:
                    Linedata = [0 for i in range(10)]
                if str(Linedata[0]).lower() == key.lower():
                    param = Linedata
            except:
                pass
    else:
        with open(origdir+"Errormessage",mode='a') as df:
            df.write("The loop is killed because"+path+" does not exist\n")
            df.flush()
        sys.exit()

    linecache.clearcache()
    return  param

#*** progdyn.conf
def confread_count(filename,key):
    param = 0 
    if os.path.exists(filename) == True:
        key = str(key)
        linenum  = sum(1 for line in open(filename))
        for line in range(linenum):
            Linedata = linecache.getline(filename,line+1).split()
            try:
                if len(Linedata) == 0 or len(Linedata) == 1:
                    Linedata = [0 for i in range(10)]
                if str(Linedata[0]).lower() == key.lower():
                    param += 1
            except:
                pass
    else:
        with open(origdir+"Errormessage",mode='a') as df:
            df.write("The loop is killed because"+path+" does not exist\n")
            df.flush()
        sys.exit()

    linecache.clearcache()
    return  param

#*** progdyn.conf
def confread_each_list(path,key):
    conffile = progdynstarterHP.conffile
    param = [-1 for i in range(10000)]
    if os.path.exists(path) == True:
        key = str(key)
        filename = path + conffile
        linenum  = sum(1 for line in open(filename))
        for line in range(linenum):
            Linedata = linecache.getline(filename,line+1).split()
            try:
                if len(Linedata) == 0 or len(Linedata) == 1:
                    Linedata = [0 for i in range(10)]
                if str(Linedata[0]).lower() == key.lower(): 
                    param[int(Linedata[1])] = Linedata[2]
            except:
                pass
    else:       
        with open(origdir+"Errormessage",mode='a') as df:
            df.write("The loop is killed because"+path+" does not exist\n")
            df.flush()
        sys.exit()
        
    linecache.clearcache()
    return  param

#*** progdyn.conf
def confread_each_list_expanded(filename,key,rownum,columnnum):
    param = [ [0 for i in range(int(rownum))] for j in range(int(columnnum))]
    countedline = 0
    if os.path.exists(filename) == True:
        key = str(key)
        linenum  = sum(1 for line in open(filename))
        for line in range(linenum):
            Linedata = linecache.getline(filename,line+1).split()
            try:
                if len(Linedata) == 0 or len(Linedata) == 1:
                    Linedata = [0 for i in range(10)]
                if str(Linedata[0]).lower() == key.lower(): 
                    for j in range(1,len(Linedata)):
                        param[countedline][j-1] = Linedata[j]
                    countedline += 1
            except:
                pass
    else:       
        with open(origdir+"Errormessage",mode='a') as df:
            df.write("The loop is killed because"+path+" does not exist\n")
            df.flush()
        sys.exit()
    #print(param)    
    linecache.clearcache()
    return  param

#*** status
def statusread(origdir,parameter,val):
    # Initialize values. These values are default.
    nogo,progress,bypassproggen,skipstart = "on","Initialized","off","forwardstart"
    isomernum,runpointnum,potE0th = 0,0,0
    totE1stpoint = 0

    # Here we make status file if it does not exist in the original directory
    if os.path.exists(origdir+"status") == False:
    #    print("make new status file")
        with open(origdir+"status",mode="w") as sw:
            sw.write("nogo "+str(nogo)+"\nprogress "+str(progress)+\
            "\nbypassproggen "+str(bypassproggen)+ "\nskipstart "+str(skipstart)+\
            "\nisomernum "+str(isomernum)+"\nrunpointnum "+str(runpointnum)+\
            "\npotE0th"+str(potE0th)+"\ntotE1stpoint"+str(totE1stpoint))
            sw.flush()
    # If the "status" file exists in the original directory, read it.
    else:
        for line in range(sum(1 for line in open(origdir+"status"))):
            Linedata = linecache.getline(origdir+"status",line+1).split()
            if len(Linedata) == 0 or len(Linedata) == 1:
                Linedata = [0 for i in range(10)]
            if str(Linedata[0]) == "nogo":
                nogo = str(Linedata[1])
            if str(Linedata[0]) == "progress":
                progress = str(Linedata[1])
            if str(Linedata[0]) == "bypassproggen":
                bypassproggen = str(Linedata[1])
            if str(Linedata[0]) == "skipstart":
                skipstart = str(Linedata[1])
            if str(Linedata[0]) == "isomernum":
                isomernum = int(Linedata[1])
            if str(Linedata[0]) == "runpointnum":
                runpointnum = int(Linedata[1])
            if str(Linedata[0]) == "potE0th":
                potE0th = float(Linedata[1])
            if str(Linedata[0]) == "totE1stpoint":
                totE1stpoint = float(Linedata[1])

        linecache.clearcache()

    # Here we change the input parameters
    if parameter == "nogo":
        nogo = val
    elif parameter == "progress":
        progress = val
    elif parameter == "skipstart":
        skipstart = val
    elif parameter == "isomernum":
        isomernum = int(val)
    elif parameter == "runpointnum":
        runpointnum = int(val)
    elif parameter == "bypassproggen":
        bypassproggen = val
    elif parameter == "potE0th":
        potE0th = int(val)
    elif parameter == "totE1stpoint":
        totE1stpoint = val

    # Write the new parameters in the "status" file
    with open(origdir+"status",mode='w') as sw:
        sw.write("nogo "+nogo+"\nprogress "+progress+"\n")
        sw.write("bypassproggen "+bypassproggen+"\nskipstart "+skipstart)
        sw.write("\nisomernum "+str(isomernum)+"\nrunpointnum "+str(runpointnum))
        sw.write("\npotE0th "+str(potE0th)+"\ntotE1stpoint "+str(totE1stpoint))
        sw.flush()

    status = {"nogo":nogo,"progress":progress,\
              "bypassproggen":bypassproggen,"skipstart":skipstart,\
              "isomernum":isomernum,"runpointnum":runpointnum,\
              "potE0th":potE0th,"totE1stpoint":totE1stpoint}
#    print("status",status)
    return status

#*** packmol output file
def pdbread(filename):
    alphabet_int = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,
                    "K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,
                    "T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}
    atom_weight = {'H':1.00783,'He':4.0026,'Li':6.941,'Be':9.012,'B':10.811,
                'C':12.,'N':14.007,'O':15.9994,'F':18.9984,'Ne':20.1797,
                'Na':22.989,'Mg':24.305,'Al':26.98154,'Si':28.0855,'P':30.9738,
                'S':32.066,'Cl':35.4527,'Ar':39.948,'K':39.0983,'Ca':40.078,
                'Sc':44.96,'Ti':47.867,'V':50.94,'Cr':51.9961,'Mn':54.938,
                'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,
                'Ga':69.723,'Ge':72.64,'As':74.9216,'Se':78.96,'Br':79.904,
                'Pd':106.42,'I':126.90447}

    linenum  = sum(1 for line in open(filename))
    geoArr     = []
    molnum     = []
    layernum   = []
    atSym      = []
    atWeight   = []
    for line in range(linenum):
        Linedata = linecache.getline(filename,line+1).split()
        if len(Linedata) < 7:
            Linedata = [ "Null" for i in range(10) ]
        try:
            if int(Linedata[1]) > 0:
                geoArr.append([float(Linedata[-4]),float(Linedata[-3]),float(Linedata[-2])])
                molnum.append(int(Linedata[-5]))
                if len(Linedata) == 8:
                    layernum.append(0)
                else:
                    layernum.append(alphabet_int[Linedata[3]])            
                atSym.append(Linedata[-1])
                atWeight.append(atom_weight[Linedata[-1]])
        except:
            pass
    geoArr = np.array(geoArr)
    molnum = np.array(molnum) - 1
    layernum = np.array(layernum) 
    atWeight = np.array(atWeight)

    return geoArr,molnum,layernum,atSym,atWeight 

#*** traj
def trajread(origdir,num_str=10,filename="traj"):
    atomnum = int(linecache.getline(origdir+filename,1).split()[0])
    linenum = sum(1 for line in open(origdir+filename))
    atoms = [ i for i in range(atomnum)]
    for atm in range(atomnum):
        coordinates_information = linecache.getline(origdir+filename,atm + 3).split()
        atoms[atm] = coordinates_information[0]
    trajs = np.zeros([num_str,atomnum,3])
    potE = np.zeros(num_str)
    for time in range(int(num_str)):
        lineinfo = linecache.getline(origdir+filename,linenum - (num_str-time)*(2+atomnum)+2).split()
        potE[time] = lineinfo[0]
        for atm in range(atomnum):
            coordinates_information = \
            linecache.getline(origdir+filename,linenum - (num_str-time)*(2+atomnum) + atm +3).split()
            for i in range(3):
                trajs[time][atm][i] = float(coordinates_information[i+1]) 

    return potE,atoms,trajs

def errorlog(origdir,comment):
    status = statusread(origdir,0,0)
    with open(origdir+"Errorlog",mode='a') as ew:
        ew.write(comment+" runpointnumber "+str(status["runpointnum"])+" ")
        date = time.localtime()
        ew.write(time.strftime("%Y-%h-%d %H:%M:%S",date)+"\n")
        ew.flush()

#*** centeratomPhase 
def centeratomPhaseread(origdir,filename="centeratomPhase"):
    modenum = int(linecache.getline(origdir+filename,1).split()[0])
    atomnum = int(linecache.getline(origdir+filename,1).split()[1])
    print(atomnum,modenum)
    amplitude = np.zeros(modenum)
    angvel = np.zeros(modenum)
    phase = np.zeros(modenum)
    mode = np.zeros([modenum,atomnum,3])
    for line in range(sum(1 for line in open(origdir+filename))):        
        lineinfo = linecache.getline(origdir+filename,line+1).split()
        if len(lineinfo) < 3:
            lineinfo.append("Null")
#        print(lineinfo)
        if lineinfo[0] == "amplitude":
            amplitude[int(lineinfo[1])] = float(lineinfo[2])
        if lineinfo[0] == "angvel":
            angvel[int(lineinfo[1])] = float(lineinfo[2])
        if lineinfo[0] == "phase":
            phase[int(lineinfo[1])] = float(lineinfo[2])
        if lineinfo[0] == "rotated_mode":
            for j in range(3):
                mode[int(lineinfo[1])][int(lineinfo[2])][j] = float(lineinfo[j+3])

    return amplitude,angvel,phase,mode

if __name__ == '__main__':
#    freqfile = sys.argv[1]
#    confreader("./")
#    geoArr,molnum,layernum,atSym,atWeight = pdbread("./solvgeo.pdb")
#    print("geoArr\n",geoArr,"\nmolnum\n",molnum,"\nlayernum\n",layernum,"\natSym\n",atSym,"\natWeight\n",atWeight)
#    print(confread_count("./","exforceset"))
#    confread_each_list_expanded("./progdyn.conf","exforceset",5,1)    
#    geoArr,atSym,atWeight = structurereader(freqfile)
#    mode,freq,force,redMass = normal_mode_reader(freqfile)
#    print("geoArr", geoArr,"atSym",atSym,"atWeight",atWeight)
    origdir = os.getcwd()
    if origdir[-1] != "/":
        origdir = origdir + "/"
#    potE,atoms,trajs = trajread(origdir,num_str=10,filename="traj")
#    val = progdynstarterHP.check_totE(origdir,10)
#    print(val)
    amplitude,angvel,phase,mode = centeratomPhaseread(origdir,filename="centeratomPhase")
    print(amplitude,angvel,phase,mode)



