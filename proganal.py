############################# quasiproganal.py ################################
#
#    Analyzing program.
#    Based on "proganal" of PROGDYN suite written by Daniel A. Singleton
#    
#    written by Hiroaki Kurouchi
#    ver 06/14/2018
#    ver 12/05/2018 Bug is fixed
#    ver 04/26/2019 The tailcheck is modified. It always returns 1 when autodamp is not 0
#    ver 02/14/2020 The tailcheck is modified to ignore termination process if autodamp > 0
#    ver 04/05/2020
#
################################################################################

import numpy as np
import gausslog
import time
import linecache
import progread
import os
import sys
import progdynstarterHP
# Define parameters
# You should put "w" as the third value for writing the values into dynfollowfile
def getparam(structure):
    param = [
             ["C1_O25",    Distance(structure,1,25),  "w"],         # Param[0]
             ["C2_O25",    Distance(structure,1,25),  "w"],         # Param[1]
             ["C1_O23",    Distance(structure,1,23),  "n"],         # Param[2]
             ["C2_O23",    Distance(structure,1,23),  "n"],         # Param[3]
             ["C1_O18",    Distance(structure,1,18),  "n"],         # Param[4]
             ["C2_O18",    Distance(structure,1,18),  "n"],         # Param[5]
             ["C1_O22",    Distance(structure,1,22),  "n"],         # Param[6]
             ["C2_O22",    Distance(structure,1,22),  "n"],         # Param[7]

             ["C1_H5",    Distance(structure,1,5),  "n"],         # Param[8]
             ["C2_H5",    Distance(structure,2,5),  "n"],         # Param[9]
             ["C10_H5",    Distance(structure,10,5),  "n"],         # Param[10]
             ["C1_H16",    Distance(structure,1,16),  "n"],         # Param[11]
             ["C2_H16",    Distance(structure,2,16),  "n"],         # Param[12]
             ["C10_H16",    Distance(structure,10,16),  "n"],         # Param[13]
             ["C1_H11",    Distance(structure,1,11),  "n"],         # Param[14]
             ["C2_H11",    Distance(structure,2,11),  "n"],         # Param[15]
             ["C10_H11",    Distance(structure,10,11),  "n"],         # Param[16]
             ["C1_H12",    Distance(structure,1,12),  "n"],         # Param[17]
             ["C2_H12",    Distance(structure,2,12),  "n"],         # Param[18]
             ["C10_H12",    Distance(structure,10,12),  "n"],         # Param[19]

             ["C7_H9",    Distance(structure,7,9),  "n"],         # Param[20]
             ["C4_H20",    Distance(structure,4,20),  "n"],         # Param[21]

           ]
    return param

# Here define Killing process rule
def process_killrule(origdir,param):
    status = progread.statusread(origdir,0,0)
    runpoint = status["runpointnum"]
    comment   = "0"

    if runpoint > 5000:
        comment = "Too many points XXXX"

    if True in [ param[i][1] < 1.5 for i in [ j*2 for j in range(4)]]:
        comment = "C1 Addition XXXX"

    if True in [ param[i][1] < 1.5 for i in [ j*2+1 for j in range(4)]]:
        comment = "C2 Addition XXXX"

    if True not in [ param[i][1] < 1.5 for i in [ 8,9,10 ]]:
        comment = "H5 Dissociated XXXX"

    if True not in [ param[i][1] < 1.5 for i in [ 11,12,13 ]]:
        comment = "H16 Dissociated XXXX"

    if True not in [ param[i][1] < 1.5 for i in [ 14,15,16 ]]:
        comment = "H11 Dissociated XXXX"

    if True not in [ param[i][1] < 1.5 for i in [ 17,18,19 ]]:
        comment = "H12 Dissociated XXXX"

    if True not in [ param[i][1] < 1.5 for i in [ 20,21 ]]:
        comment = "Returned to norbornene  XXXX"


    return comment

################################################################################
################## Be careful when you edit the code below #####################
################################################################################
def proganal(inputfilename,outputfilename,origdir,structure=0):
#    structure,Atoms,Atomweight = gausslog.structurereader(inputfilename)
    try:
        if structure == 0:
            structure = gausslog.inputstructurereader(inputfilename)
    except:
        pass

    if np.sum(structure) == 0:
        try:
            structure = gausslog.trajstructurereader(origdir+"traj",origdir+"olddynrun",1)
        except:
            print("No traj file")
    param = getparam(structure)
    write_dynfollowfile(param,inputfilename,outputfilename,origdir)
    if tailcheck(origdir+"Echeck",origdir) == 0:
        with open(outputfilename,mode='a') as df:
            df.write("XXXX bad total Energy\n")
            df.flush()

def write_dynfollowfile(param,inputfilename,outputfilename,origdir):
    title1,title2,title3,title4 = progread.confread_each(origdir,"title","you")[1],\
                                  progread.confread_each(origdir,"title","need")[2],\
                                  progread.confread_each(origdir,"title","a")[3],\
                                  progread.confread_each(origdir,"title",progdynstarterHP.conffile)[4]
    status = progread.statusread(origdir,0,0)
    comment = process_killrule(origdir,param)
    with open(origdir+"dynfollowfile",mode='a') as df:
        df.write(str(title1)+" "+str(title2)+" "+str(title3)+" "+str(title4)+" "+\
                str(status["runpointnum"])+" isomernum "+str(status["isomernum"])+" ")
        for i in range(len(param)):
            try:
                if str(param[i][2]) == "w":
                    df.write(str(param[i][0]) + "{: .3f}".format(param[i][1])+" ")
            except:
                pass
        if comment != "0":
            df.write(comment+"\n")
        date = time.localtime()
        df.write(time.strftime("%Y-%h-%d %H:%M:%S",date)+"\n")
        df.flush()

def tailcheck(filename,origdir,readlinenum=2):
    val = 1
    if os.path.exists(filename) == True :
#        print("dotailcheck")
        linenum = sum(1 for i in open(filename))
        lastline = linecache.getline(filename,linenum).split()
        Nlastline = "null"
        try:
            Nlastline = linecache.getline(filename,linenum-1).split()
        except:
            pass

        if readlinenum == 1 and "XXXX" in lastline:
            val = 0
        if readlinenum == 2 and "XXXX" in lastline or "XXXX" in Nlastline:
            val = 0
    linecache.clearcache()
    if int(progread.confread_each(origdir,"autodamp",0)[1]) != 0:
        val = 1
    print("tailcheckresult ",val)
    return val

def donecheck(filename):
    val = 0
    if os.path.exists(filename) == True:
        linenum = sum(1 for i in open(filename))
        lastline = linecache.getline(filename,linenum).split()
        Nlastline = "null"
        try:
            Nlastline = linecache.getline(filename,linenum).split()
        except:
            pass
        if ("Normal" in lastline) or ("Normal" in Nlastline):
            val = 1 
    linecache.clearcache()
    print("donecheckresult ",val)
    return val

def mopacdonecheck(filename):
    val = 0
    if os.path.exists(filename) == True:
        linenum = sum(1 for i in open(filename))
        for line in range(linenum-100,linenum):
            lineinfo = linecache.getline(filename,line).split()
            try:
                if "NORMALLY" in lineinfo:
                    val = 1
            except:
                pass
    linecache.clearcache()
    return val

def Distance(structure,atom1,atom2):
    return np.linalg.norm(structure[atom1-1]-structure[atom2-1]) 

def Angle(structure,atom1,atom2,atom3):
    v1,v2 = structure[atom1-1]-structure[atom2-1],structure[atom3-1]-structure[atom2-1]
    return np.rad2deg(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))

def Dihedral(structure,atom1,atom2,atom3,atom4):
    # set atom2 to original point
    coord1 = structure[atom1-1]-structure[atom2-1]
    coord3 = structure[atom3-1]-structure[atom2-1]
    coord4 = structure[atom4-1]-structure[atom2-1]
    # set atom 3 to X axis
    r_xyz = np.linalg.norm(coord3)
    r_xy  = np.linalg.norm(coord3[0:2])
    sint  = coord3[1] / r_xy
    cost  = coord3[0] / r_xy
    sinp  = r_xy    / r_xyz
    cosp  = coord3[2] / r_xyz
    rot_mat_yx = np.array([[ cost, sint, 0],
                           [-sint, cost, 0],
                           [    0,    0, 1]])
    rot_mat_zx = np.array([[ sinp, 0, cosp],
                           [    0, 1,    0],
                           [-cosp, 0, sinp]])
    coord1 = np.dot(rot_mat_zx,np.dot(rot_mat_yx,coord1.T)).T
    coord4 = np.dot(rot_mat_zx,np.dot(rot_mat_yx,coord4.T)).T
    coord1[0] = coord4[0] = 0.0
    dihedral = (np.arccos(np.dot(coord1[1:],coord4[1:])
                /(np.linalg.norm(coord1[1:])*np.linalg.norm(coord4[1:])))
                * 180 * np.sign(np.cross(coord1,coord4)[0]) /np.pi )
    return dihedral

if __name__ == "__main__":
    filename=sys.argv[1]
#    proganal("./freqinHP","./dynfollowfile","./")
    print(mopacdonecheck(filename))


