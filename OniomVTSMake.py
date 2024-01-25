############################### OniomVTSMake.py #################################
#
#   A program to make ONIOM frequency calculation input file for SPTS sampling 
#
#    written by Hiroaki Kurouchi
#    ver 1.0 04/15/2019 standard process was implemented
#    ver 1.1 02/17/2020 Bug was fixed
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
def oniomgen(origdir,filename="spts.out"):
    if os.path.exists("geoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir)
    elif os.path.exists("solvgeoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir,"solvgeoPlusVel")
    structure,Atoms,Atomweight = gausslog.structurereader(filename)

    inputwrite(origdir,structure,atSym,"sptsircforward.gjf","for")
    inputwrite(origdir,structure,atSym,"sptsircreverse.gjf","rev")

def freqgen(origdir,outputfile1,outputfile2,inputfilename="sptsVTS.gjf"):
    readstrnum = 7 
    if os.path.exists("geoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir)
    elif os.path.exists("solvgeoPlusVel") == True:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE \
               = progread.geoPlusVelread(origdir,"solvgeoPlusVel")

    isonum = progread.confread_count("progdyn.conf","isotope")
    isotopes =  progread.confread_each_list_expanded("progdyn.conf","isotope",isonum,2)
    for i in range(len(isotopes)):
        atWeight[int(isotopes[i][0])-1] = float(isotopes[i][1])
    structures_for_pre = gausslog.getinputstructures(outputfile1,readstrnum)
    structures_rev_pre = gausslog.getinputstructures(outputfile2,readstrnum)
    structures_for = []
    structures_rev = []
    for i in range(readstrnum):
        if not np.sum(structures_for_pre[i]**2,axis=1)[0] == 0:
            structures_for.append(structures_for_pre[i])
        if not np.sum(structures_rev_pre[i]**2,axis=1)[0] == 0:
            structures_rev.append(structures_rev_pre[i])
    structure = structures_rev[::-1] + structures_for[1:]
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
        for strnum in range(len(geoArr)):
            gv.write("#p oniom("+highmethod+":"+method+")  " )
            if int(isonum) == 0:
                gv.write("freq=hpmodes")
            else:
                gv.write("freq=(hpmodes,readiso)")
            gv.write("\n\n"+ "VTS calc frequencies \n\n")
            gv.write(str(charge)+" "+str(multiplicity)+" "+str(highcharge)+" "+str(highmulti)+" "+"\n")
            for i in range(len(geoArr[strnum])):
                if i < highatomnum:
                    gv.write(" {:2}".format(atSym[i])+" 0 {: .7f} {: .7f} {: .7f} H".format(\
                             geoArr[strnum][i][0],geoArr[strnum][i][1],geoArr[strnum][i][2])+"\n")
                else:
                    gv.write(" {:2}".format(atSym[i])+" -1 {: .7f} {: .7f} {: .7f} L".format(\
                             geoArr[strnum][i][0],geoArr[strnum][i][1],geoArr[strnum][i][2])+"\n")
            gv.write("\n")
            if int(isonum) > 0:
                gv.write(str(temperature)+" 1.0\n")
                for atomnum in range(len(atWeight)):
                    gv.write(str(atWeight[atomnum])+"\n")
            gv.write("\n--Link1--\n")

def inputwrite(origdir,geoArr,atSym,filename,direction):
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
        if direction == "for":
            gv.write("#p oniom("+highmethod+":"+method+") \n"+\
                     "irc=(forward,recorrect=never,stepsize=3,calcfc) scf=xqc \n\n"+\
                     "IRCcalculation forward \n\n")
        elif direction == "rev":
            gv.write("#p oniom("+highmethod+":"+method+") \n"+\
                     "irc=(reverse,recorrect=never,stepsize=3,calcfc)  scf=xqc \n\n"+\
                     "IRCcalculation reverse\n\n")
        gv.write(str(charge)+" "+str(multiplicity)+" "+str(highcharge)+" "+str(highmulti)+" "+"\n")
        for i in range(len(geoArr)):
            if i < highatomnum:            
                gv.write(" {:2}".format(atSym[i])+" 0 {: .7f} {: .7f} {: .7f} H".format(\
                         geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
            else:
                gv.write(" {:2}".format(atSym[i])+" -1 {: .7f} {: .7f} {: .7f} L".format(\
                         geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
        gv.write("\n")


if __name__ == "__main__":
    origdir = os.getcwd()  
    pdbfile  = progdynstarterHP.pdbfile
    if origdir[-1] != "/":
        origdir = origdir + "/"
    if os.path.exists(origdir+"solvtraj") == True and os.path.exists(origdir+"spts.out") == True\
        and os.path.exists(origdir+"sptsircforward.out") == False:
        print("making oniom input: use g16sub -q PN -walltime 48:00:00 -np 18 -j core -mem 86000MB sptsircforward.gjf")
        print("making oniom input: use g16sub -q PN -walltime 48:00:00 -np 18 -j core -mem 86000MB sptsircreverse.gjf")
        oniomgen(origdir,"spts.out")
    elif os.path.exists(origdir+"sptsircforward.out") == True:
        freqgen(origdir,"sptsircforward.out","sptsircreverse.out",inputfilename="spVTS.gjf")
        print("making oniom input: use g16sub -q PN -walltime 48:00:00 -np 18 -j core -mem 86000MB spVTS.gjf")
    elif os.path.exists(origdir+"spts.out") == False:
        print("Be careful, spts TS calculation is NOT done") 



