############################### OniomTSMake.py #################################
#
#   A program to make ONIOM frequency calculation input file for SPTS sampling 
#
#    written by Hiroaki Kurouchi
#    ver 11/22/2018 standard process was implemented
#    ver 04/16/2019 Modified for readiso option
#    ver 02/17/2020 Bug was fixed
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

#Define Constants
conver1=4.184E26 #dividing by this converts amu angs^2 /s^2 to kcal/mol
c=29979245800; h=6.626075E-34; avNum=6.0221415E23
RgasK=0.00198588; RgasJ=8.31447

# Make solventgeoPlusVel file so as to use for geoPlusVel
def oniomgen(origdir,filename,key=0):
    if os.path.exists("geoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir)
    elif os.path.exists("solvgeoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir,"solvgeoPlusVel")
    atomnum    = int(linecache.getline(origdir+filename,1).split()[0])
    oldArr     = np.zeros([atomnum,3])
    linenum    = sum(1 for line in open(origdir+filename))
    num_of_str = int(linenum / (atomnum + 2))
    for i in range(atomnum):
        lineinfo = linecache.getline(origdir+filename,(num_of_str-2) * (atomnum+2) + i + 3).split()
        oldArr[i][0],oldArr[i][1],oldArr[i][2] = float(lineinfo[1]),float(lineinfo[2]),float(lineinfo[3])

    if key == 0:
        inputwrite(origdir,oldArr,atSym,"spts.gjf")
    elif key == 0 or key == 1:
        inputwrite2(origdir,geoArr,atSym,"spts_shell.gjf")

    ONIOM  = int(progread.confread_each(origdir,"ONIOM",0)[1])
    spts   = int(progread.confread_each(origdir,"spts",0)[1])
    autodamp   = int(progread.confread_each(origdir,"autodamp",0)[1])
    if ONIOM == 0:
        print("ONIOM is now 0. Change the value.")
    if spts == 0:
        print("spts is now 0. Change the value.")
    if autodamp != 0:
        print("autodamp is not 0. Check it if you don't want to kill the trajectory before reaction")


def inputwrite(origdir,geoArr,atSym,filename="spts.gjf"):
    # read progdyn.conf
    charge        = int(progread.confread_each(origdir,"charge",0)[1])
    multiplicity  = int(progread.confread_each(origdir,"multiplicity",1)[1])
    method        = progread.confread_each(origdir,"method","HF/3-21G")[1]
    highcharge    = int(progread.confread_each(origdir,"highcharge",0)[1])
    highmulti     = int(progread.confread_each(origdir,"highmulti",0)[1])
    highatomnum   = int(progread.confread_each(origdir,"highatomnum",0)[1])
    highmethod    = progread.confread_each(origdir,"highmethod","HF/3-21G")[1]

    prog          = progread.confread_each(origdir,"prog","gaussian")[1]
    highprog      = progread.confread_each(origdir,"highprog","gaussian")[1]

    if prog == "mopac":
        if method == "pm7" or method == "PM7":
            method = "PM7MOPAC"

    # Here we generate ONIOM input file
    with open(origdir+filename,mode='w') as gv:
        gv.write("#p oniom("+highmethod+":"+method+") \n opt=(ts,noeigentest,recalcfc=5) freq=hpmodes scf=xqc \n\n"+\
                 "TScalculation \n\n")
        gv.write(str(charge)+" "+str(multiplicity)+" "+str(highcharge)+" "+str(highmulti)+" "+"\n")
        for i in range(len(geoArr)):
            if i < highatomnum:            
                gv.write(" {:2}".format(atSym[i])+" 0 {: .7f} {: .7f} {: .7f} H".format(\
                         geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
            else:
                gv.write(" {:2}".format(atSym[i])+" -1 {: .7f} {: .7f} {: .7f} L".format(\
                         geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
        gv.write("\n")

def inputwrite2(origdir,geoArr,atSym,filename="spts_shell.gjf"):
    # read progdyn.conf
    charge        = int(progread.confread_each(origdir,"charge",0)[1])
    multiplicity  = int(progread.confread_each(origdir,"multiplicity",1)[1])
    method        = progread.confread_each(origdir,"method","HF/3-21G")[1]
    highatomnum   = int(progread.confread_each(origdir,"highatomnum",0)[1])

    prog          = progread.confread_each(origdir,"prog","gaussian")[1]

    if prog == "mopac":
        if method == "pm7" or method == "PM7":
            method = "PM7MOPAC"

    # Here we generate ONIOM shell input file
    with open(origdir+filename,mode='w') as gv:
        gv.write("#p "+method+"   scf=xqc \n\n"+\
                 "TS shell calculation \n\n")
        gv.write(str(charge)+" "+str(multiplicity)+"\n")
        for i in range(highatomnum,len(geoArr)):
            gv.write(" {:2}".format(atSym[i])+" {: .7f} {: .7f} {: .7f} ".format(\
                     geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
        gv.write("\n")

def freqgen(origdir,outputfile,inputfilename="spts.gjf"):
    if os.path.exists("geoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir)
    elif os.path.exists("solvgeoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir,"solvgeoPlusVel")

    # Here isotopes are read as a list [[atomnumber1,weight1],[atomnumber2,weight2]...]
    isonum = progread.confread_count("progdyn.conf","isotope")
    isotopes =  progread.confread_each_list_expanded("progdyn.conf","isotope",isonum,2)
    # Because atomnumbers starts from 0, the read atom number from progdyn.conf should be subtracted.
    for i in range(len(isotopes)):
        if abs(atWeight[int(isotopes[i][0])-1] - float(isotopes[i][1])) > 1.5:
            print("!!! Be careful, the atomic isotope weight is quite different from the original one !!!")
        print("Atom number: ",isotopes[i][0]," Original weight: ",atWeight[int(isotopes[i][0])-1]," New weight: ",float(isotopes[i][1]))
        atWeight[int(isotopes[i][0])-1] = float(isotopes[i][1])
    structure,Atoms,Atomweight = gausslog.structurereader(outputfile)
    inputwrite_freq(origdir,structure,atSym,atWeight,inputfilename)

def inputwrite_freq(origdir,geoArr,atSym,atWeight,filename):
    # read progdyn.conf
    charge        = int(progread.confread_each(origdir,"charge",0)[1])
    multiplicity  = int(progread.confread_each(origdir,"multiplicity",1)[1])
    method        = progread.confread_each(origdir,"method","HF/3-21G")[1]
    highcharge    = int(progread.confread_each(origdir,"highcharge",0)[1])
    highmulti     = int(progread.confread_each(origdir,"highmulti",0)[1])
    highatomnum   = int(progread.confread_each(origdir,"highatomnum",0)[1])
    highmethod    = progread.confread_each(origdir,"highmethod","HF/3-21G")[1]

    prog          = progread.confread_each(origdir,"prog","gaussian")[1]
    highprog      = progread.confread_each(origdir,"highprog","gaussian")[1]
    isonum = progread.confread_count("progdyn.conf","isotope")
    temperature   = float(progread.confread_each(origdir,"temperature",298.15)[1])

    if prog == "mopac":
        if method == "pm7" or method == "PM7":
            method = "PM7MOPAC"

    # Here we generate ONIOM input file
    with open(origdir+filename,mode='w') as gv:
        gv.write("#p oniom("+highmethod+":"+method+")  " )
        if int(isonum) == 0:
            gv.write("freq=hpmodes")
        else:
            gv.write("freq=(hpmodes,readiso)")
        gv.write("\n\n"+ "TS calc frequencies \n\n")
        gv.write(str(charge)+" "+str(multiplicity)+" "+str(highcharge)+" "+str(highmulti)+" "+"\n")
        for i in range(len(geoArr)):
            if i < highatomnum:
                gv.write(" {:2}".format(atSym[i])+" 0 {: .7f} {: .7f} {: .7f} H".format(\
                         geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
            else:
                gv.write(" {:2}".format(atSym[i])+" -1 {: .7f} {: .7f} {: .7f} L".format(\
                         geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
        gv.write("\n")
        if int(isonum) > 0:
            gv.write(str(temperature)+" 1.0\n")
            for atomnum in range(len(atWeight)):
                gv.write(str(atWeight[atomnum])+"\n")
        gv.write("\n")

if __name__ == "__main__":
    origdir = os.getcwd()  
    pdbfile  = progdynstarterHP.pdbfile
    if origdir[-1] != "/":
        origdir = origdir + "/"
    if os.path.exists(origdir+"solvtraj") == True and os.path.exists(origdir+"spts.gjf") == False:
        print("making ONIOM TS input file: use g16sub -q PN -walltime 48:00:00 -np 18 -j core -mem 86000MB spts.gjf")
        oniomgen(origdir,"solvtraj")
    if os.path.exists(origdir+"spts_shell.gjf") == False:
        print("spts_shell.gjf is now created")
        oniomgen(origdir,"solvtraj",key=1)
    if os.path.exists(origdir+"spts.out") == True:
        print("Be careful, spts calculation is already done") 
        if progread.confread_count("progdyn.conf","isotope") > 0:
            print("The older spts calculation input and output files are now renamed.")
            if os.path.exists(origdir+"spts_geometryopt.out") == False:
                os.rename(origdir+"spts.out",origdir+"spts_geometryopt.out")
            if os.path.exists(origdir+"spts.gjf") == True and os.path.exists(origdir+"spts_geometryopt.gjf") == False:
                os.rename(origdir+"spts.gjf",origdir+"spts_geometryopt.gjf")
                freqgen(origdir,"spts_geometryopt.out",inputfilename="spts.gjf")
                print("making oniom input: use g16sub -q PN -walltime 48:00:00 -np 18 -j core -mem 86000MB spts.gjf")




