######################## (quasi)progdynstarterHP.py ######################################
#
#   Main program for running trajectory calculation. 
#   This code is based on progdynstarterHP, written by Daniel A. Singleton
#   Original paper: D.Singleton et.al. J. Am. Chem. Soc. 2003, 125, 1176-1177.
#
#   For python3.4 or later
#
#   written by Hiroaki Kurouchi
#   ver 06/17/2018 
#   ver 06/20/2018 PROGDYNONIOM method is implemented
#   ver 08/09/2018 The program is reorganized for Fail-safe restarting.
#                  The format of "status: goingwell" is changed to "status: progress".
#                  This version is only compatible with progread ver1.2 or later
#   ver 08/20/2018 status = "g16Running" is added for fixing bug
#   ver 08/27/2018 Bug is fixed for starting new isomer after failure of Echeck
#   ver 09/05/2018 force_calculation_run is throughly modified.
#   ver 09/08/2018 Modified for MOPAC2016 calculation 
#   ver 09/19/2018 Bug is fixed and the structure of this program is modifiled
#   ver 01/23/2019 Fail-safe force calculation is implemented
#   ver 01/28/2019 check_totE was implemented
#   ver 02/26/2020 minor revision
#   ver 04/05/2020 minor revision
#
#########################################################################################

# Standard library
import linecache
import os
import sys
import shutil
import subprocess
import numpy as np
import time

# Original modules
import gausslog
import progread
import proggenHP
import proganal
import progcommake
#import testg16

### input paths and filenames ###
# The path must end with slash "/"
g16root      = "/local/apl/lx/g16a03/g16/"                  # CHECK HERE
mopacroot    = "/home/users/to1/MOPAC/"                     # CHECK HERE
mopacprogram = "MOPAC2016.exe"
programdir   = "/home/users/to1/binall800/"                 # CHECK HERE
freqfile     = programdir + "freqinHP"
logfile      = "hirolog"
forceoption  = "force scf=(xqc,maxconv=35,fulllinear,nosym)"
forceoption2 = "force scf=(yqc,maxconv=10,fulllinear,nosym)"
forceoption3 = "force scf=(xqc,maxconv=5,fulllinear,vtl,nosym)"
forceoption4 = "force scf=(fermi,vtl,nosym)"
forceoption5 = "force scf=(xqc,maxconv=5,fulllinear,vtl,nosym,vshift=1000)"
g16env       = g16root + "bsd/g16.profile"        # CHECK HERE

mopacforceoption = "XYZ GRADIENTS 1SCF PRTXYZ"
mopacforceoption2 = "XYZ GRADIENTS 1SCF PRTXYZ"
mopacforceoption3 = "XYZ GRADIENTS 1SCF PRTXYZ"

conffile     = "progdyn.conf"
pdbfile      = "solvgeo.pdb"

inputcomfile    = "g16.com" 
outputlogfile   = "g16.log"
mopacfile       = "mopac2016"


################################ Be careful to edit code below ##############################
### check the path input ###
subprocess.call("source " + g16env,shell=True)
#try:
#    subprocess.call("export " + mopacroot,shell=True)
#except:
#    pass


# Because this program is not for NMR calculation, tcheck is not checked

### L1 --- Main loop for starting, propageting trajectories ###
###
# In this loop, progress value can be input as Initialized, Incrementrunpointnum, g16Prepared,
# g16Done, g16Filesrenamed or Trajendjobdone.  Otherwise the loop stops without propagation.

### The "progress" value leads you to start the calculation from specified points for Fail-safe restart.
# States of "progress":
# Initialized          --- Begin new trajectory or do reversestart
# Incrementrunpointnum --- Skip L2 initialization and starts from L3 runpointnum increment
# g16Prepared          --- Start from g16 calculation on L3 loop
# g16Running           --- Start from g16 calculation skipping progcommake 
# g16Done              --- Start from copying the result on L3 loop
# g16Filesrenamed      --- Start from checking whether trajectory calculation is done or not
#                          from this section, the progress value trifurcates 
#                          to Incrementrunpoinum(for next point of the trajectory), 
#                          Trajendjobdone(for next isomer) or Initialized(for next isomer or reverse)
# Trajendjobdone       --- Begin new isomer 

def L1_firstloop(origdir,scratchdir):
    print("The 1st loop start")
    while True:
        ### Here we start L2 loop ###
        L2_secondloop(origdir,scratchdir)

        status = progread.statusread(origdir,0,0)
        if status["progress"] in ["Trajendjobdone","Echeckfailed"]:
            if int(progread.confread_each(origdir,"trajloop",1)[1]) == 0 \
                               and status["progress"] == "Trajendjobdone":
                status = progread.statusread(origdir,"progress","Initialized")
                status = progread.statusread(origdir,"skipstart","forwardstart")
                break

            status = progread.statusread(origdir,"progress","Initialized")
            status = progread.statusread(origdir,"skipstart","forwardstart")

        else:
            print("get out from the loop L1")
            break

### L2 --- Start trajectory calculation and then start propagation ###
### In this loop, progress value can be input as Initialized, 
def L2_secondloop(origdir,scratchdir):
    print("The 2nd loop start")
    diag = int(progread.confread_each(origdir,"diagnostics",0)[1])
    while True:
        breakflag = 0
        status = progread.statusread(origdir,0,0)

        ### Prepare for starting trajectory ###
        if status["progress"] == "Initialized":
            ### Forwardstart process ###
            if status["skipstart"] == "forwardstart":
                breakflag = start_function(origdir,scratchdir)
                if breakflag == 1:
                    status = progread.statusread(origdir,"progress","Echeckfailed")
                    if diag == 1:
                        with open (origdir+"diagnostics",mode='a') as dw:
                            dw.write("forwardstart failed\n")
                            dw.flush()
                    break
                else:
                    status = progread.statusread(origdir,"skipstart","forward")
                    if diag == 1:
                        with open (origdir+"diagnostics",mode='a') as dw:
                            dw.write("forwardstart successed\n")
                            dw.flush()

            ### Reverserestart process ###
            if status["skipstart"] == "reverserestart":
                breakflag = reversestart_function(origdir,scratchdir)
                if breakflag == 1:
                    status = progread.statusread(origdir,"progress","Echeckfailed")
                    break
                else:
                    status = progread.statusread(origdir,"skipstart","reverse")
            status = progread.statusread(origdir,"progress","Incrementrunpointnum")

        ### Then start loop L3 ###
        L3_thirdloop(origdir,scratchdir)

        # check stop this trajectory or not
        status = progread.statusread(origdir,0,0)
        if status["progress"] == "Initialized":
            print("starting a new point for new direction")
        elif status["progress"] == "Trajendjobdone":
            break

### L3 --- Main loop for propagation ###
def L3_thirdloop(origdir,scratchdir):
    print("The 3rd loop start")
    status = progread.statusread(origdir,0,0)
    if status["progress"] not in ["Incrementrunpointnum","g16Prepared","g16Running","g16Done","g16Filesrenamed"]:
        print("something is wrong in progress value")

    else:
        while True:
            status = progread.statusread(origdir,0,0)
            # Increment runpointnum. The order of this increment is different from that of original PROGDYN
            if status["progress"] == "Incrementrunpointnum":
                runpointnum = 1 + status["runpointnum"]
                status = progread.statusread(origdir,"runpointnum",runpointnum)
                status = progread.statusread(origdir,"progress","g16Prepared")

            # Create g16.com file and run gaussian16
            if status["progress"] == "g16Prepared" or status["progress"] == "g16Running":
                force_calculation_run(origdir,scratchdir)  #progress -> g16Done

            # Calculate total energy and compare with that of original point
            totEdiff = float(progread.confread_each(origdir,"totEdiff",0)[1])
            if (status["totE1stpoint"] != 0) and (totEdiff != 0) :
                status = progread.statusread(origdir,0,0)
                if status["runpointnum"] > 11:
                    totecheck = check_totE(origdir)
                    if totecheck == -1:
                        trajendjob(origdir,word="totEfail")
                        break

            status = progread.statusread(origdir,0,0)            
            if status["progress"] == "g16Fail":
                trajendjob(origdir,word="abort")
                break

            # Move and rename g16 calculation results
            # condition branch is written redundantly for safe restart
            status = progread.statusread(origdir,0,0)
            if status["progress"] == "g16Done":
                proganal.proganal(scratchdir + outputlogfile,origdir+"dynfollowfile",origdir)
                if os.path.exists(origdir + "olddynrun3") == True:
                    os.remove(origdir + "olddynrun3")
                if os.path.exists(origdir + "olddynrun2") == True\
                   and os.path.exists(origdir + "olddynrun") == True:
                    os.rename(origdir + "olddynrun2",origdir + "olddynrun3")
                if os.path.exists(origdir + "olddynrun") == True\
                   and os.path.exists(scratchdir + outputlogfile) == True:
                    os.rename(origdir + "olddynrun",origdir + "olddynrun2")
                if os.path.exists(scratchdir + outputlogfile) == True:
                    shutil.copy(scratchdir + outputlogfile,origdir + "olddynrun")
                status = progread.statusread(origdir,"progress","g16Filesrenamed")

            # Check the trajectory calculation results. Here the calculation branches. 
            if status["progress"] == "g16Filesrenamed":
                if proganal.tailcheck(origdir + "dynfollowfile",origdir) == 0:
                    reversetraj = progread.confread_each(origdir,"reversetraj",0)[1]
                    if reversetraj == "true" or reversetraj == "1":
                        status = progread.statusread(origdir,0,0)
                        if not status["skipstart"] == "forward":
                            trajendjob(origdir) #progress -> Trajendjobdone
                        if status["skipstart"] == "forward":
                            status = progread.statusread(origdir,"skipstart","reverserestart")
                            status = progread.statusread(origdir,"progress","Initialized")
                        break
                    else:
                        trajendjob(origdir) #progress -> Trajendjobdone
                        break
                else:
                    print("traj continue")
                    status = progread.statusread(origdir,"progress","Incrementrunpointnum")

def start_function(origdir,scratchdir): 
    etolerance   = int(progread.confread_each(origdir,"etolerance",7777777)[1])
    ### Post processing of the previous run
    breakflag = 0
    status = progread.statusread(origdir,0,0)
    runpointnum = status["runpointnum"] 
    diag = int(progread.confread_each(origdir,"diagnostics",0)[1])
    with open(origdir+"dynfollowfile",mode='a') as df:
        if runpointnum == 1:
            if int(status["isomernum"]) > 0:
                df.write("X did not complete first point so new isomer started\n")
            if os.path.exists(origdir + "traj"):
                os.remove(origdir + "traj")
        elif runpointnum == 2:
            df.write("X did not complete second point so new isomer started\n")
            if os.path.exists(origdir + "traj"):
                os.remove(origdir + "traj")
        elif runpointnum == 3:
            df.write("X did not complete third point so new isomer started\n")
            if os.path.exists(origdir + "traj"):
                os.remove(origdir + "traj")
        df.flush()
    status = progread.statusread(origdir,"runpointnum",1)

    ### Start New trajectory
    if status["bypassproggen"] == "on":
        print("proggen is skipped and the prepared geoPlusVel was read")
        with open(origdir+"dynfollowfile",mode='a') as df:
            df.write("proggen is skipped and the prepared geoPlusVel was read \n")
            df.flush()

        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,\
        potentialE = progread.geoPlusVelread(origdir)
    else:
        geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,\
        potentialE = proggenHP.proggenHP(programdir,origdir,freqfile)

    status = progread.statusread(origdir,"potE0th",potentialE)
    if diag == 1:
        with open (origdir+"diagnostics",mode='a') as dw:
            dw.write("proggen is finished\n")
            dw.flush()

    # Check and Create isomernumber file if it is the first loop
    status = progread.statusread(origdir,"isomernum",status["isomernum"]+1)
    if os.path.exists(origdir + inputcomfile):
        os.remove(origdir + inputcomfile)

    # Make the 1st point and calculate force and potential
    if diag == 1:
        with open (origdir+"diagnostics",mode='a') as dw:
            dw.write("1stpoint is successfully created\n")
            dw.flush()

    with open(origdir+"geoRecord",mode='a') as logwrite:
        logwrite.write(str(status["isomernum"])+" ----trajectory isomer number----\n")
        linenum  = sum(1 for line in open(origdir+"geoPlusVel"))
        for i in range(linenum):
            logwrite.write(linecache.getline(origdir+"geoPlusVel",i+1))
        logwrite.flush()
    linecache.clearcache()

    force_calculation_run(origdir,scratchdir,runpoint=1)
    if proganal.donecheck(scratchdir+outputlogfile) == 1:
        shutil.copy(scratchdir+outputlogfile,origdir+"olddynrun2")
    else:
        shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
        breakflag = 1
        return breakflag

    # Let's do Echeck
    forceArr,newpotentialE = gausslog.forcereader(origdir+"olddynrun2")
    newPotentialEK = (newpotentialE - potentialE) * 627.509
    if status["skipstart"] == "forwardstart":
        with open(origdir+"Echeck",mode='a') as ec:
            ec.write("trajectory #"+str(status["isomernum"])+"\n")
            ec.write("point 1 potential E= "+ "{: .3f}".format(newPotentialEK)+\
                     " point 1 kinetic E= "+"{: .3f}".format(KEinitmodes)+"  Total="+\
                     "{: .3f}".format(newPotentialEK+KEinitmodes)+"\n")
            ec.write("desired total energy= "+ "{: .3f}".format(desiredModeEnK)+"\n")
            status = progread.statusread(origdir,"totE1stpoint",newPotentialEK+KEinitmodes)
            if etolerance  != 0:
                if (newPotentialEK + KEinitmodes) > (desiredModeEnK + etolerance) or\
                    (newPotentialEK + KEinitmodes) < (desiredModeEnK - etolerance) :
                    ec.write("XXXX bad total Energy \n")
            ec.flush()

    # before running gaussian16, check energy
    proganal.proganal(scratchdir + outputlogfile,origdir + "dynfollowfile",origdir,structure=geoArr)
    if proganal.tailcheck(origdir + "dynfollowfile",origdir,1) == 0:
        if os.path.exists(origdir+"traj"):
            os.remove(origdir+"traj") 
        breakflag = 1
        return breakflag

    # Then make runpoint 2 and start trajectory calculation
    status = progread.statusread(origdir,"runpointnum",2)
    force_calculation_run(origdir,scratchdir,runpoint=2)
    if proganal.donecheck(scratchdir+outputlogfile) == 1:
        shutil.copy(scratchdir+outputlogfile,origdir+"olddynrun")
        proganal.proganal(scratchdir + outputlogfile,origdir + "dynfollowfile",origdir)
    else:
        shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
        breakflag = 1
        return breakflag

    return breakflag

def reversestart_function(origdir,scratchdir):  # B4 section in the original program
    breakflag = 0
    status = progread.statusread(origdir,"runpointnum",1)
    # Make the 1st point and calculate force and potential
    force_calculation_run(origdir,scratchdir,runpoint=1)
    if proganal.donecheck(scratchdir+outputlogfile) == 1:
        shutil.copy(scratchdir+outputlogfile,origdir+"olddynrun2")
    else:
        shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
        breakflag = 1
        return breakflag

    status = progread.statusread(origdir,"runpointnum",2)
    proganal.proganal(scratchdir + outputlogfile,origdir + "dynfollowfile",origdir)
    force_calculation_run(origdir,scratchdir,runpoint=2)
    if proganal.donecheck(scratchdir+outputlogfile) == 1:
        shutil.copy(scratchdir+outputlogfile,origdir+"olddynrun")
        proganal.proganal(scratchdir + outputlogfile,origdir + "dynfollowfile",origdir)
    else:
        shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
        breakflag = 1
        return breakflag

    return breakflag

def trajendjob(origdir,word="end"):
    print("trajendjob start")
    if word == "abort":
        with open(origdir+"dynfollowfile",mode='a') as df:
            df.write("SCF calculation failed and the trajectory was aborted, SCFfail\n")
            df.flush()
    if word == "totEfail":
        with open(origdir+"dynfollowfile",mode='a') as df:
            df.write("SCF calculation failed and the trajectory was aborted, totEfail\n")
            df.flush()    

    status = progread.statusread(origdir,"skipstart","forwardstart")
    if os.path.exists(origdir + "geoPlusVel") == True:
        os.remove(origdir + "geoPlusVel")
    if os.path.exists(origdir + "olddynrun") == True:
        os.remove(origdir + "olddynrun")
    if os.path.exists(origdir + "olddynrun2") == True:
        os.remove(origdir + "olddynrun2")
    if os.path.exists(origdir + "olddynrun3") == True:
        os.remove(origdir + "olddynrun3")
    isomernumber = status["isomernum"]
    if word == "end":
        os.rename(origdir + "traj",origdir + "traj" + str(isomernumber))
    if os.path.exists(origdir + "thermoinfo") == True:
        os.rename(origdir + "thermoinfo",origdir + "thermoinfo" + str(isomernumber))
    if word == "abort":
        os.rename(origdir + "traj",origdir + "aborteddyn" + str(isomernumber))
    if word == "totEfail":
        os.rename(origdir + "traj",origdir + "aborteddyn_totEfail" + str(isomernumber))
    status = progread.statusread(origdir,"progress","Trajendjobdone")

def check_totE(origdir,num_str=10):
    status = progread.statusread(origdir,0,0)
    potE,atoms,traj = progread.trajread(origdir,num_str=10,filename="traj")
    timestep      = float(progread.confread_each(origdir,"timestep",1E-15)[1])
    geoArr,velArr,atSym,weight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE\
    = progread.geoPlusVelread(origdir)
    totEdiff = float(progread.confread_each(origdir,"totEdiff",0)[1])
    potE_diff = 627.51 * (potE[1:-1] - potentialE)
    traj_diff = traj[2:] - traj[:-2]
    conver1 = 4.184E26
    KEnew = np.zeros(num_str-2)
    TotE = np.zeros(num_str-2) 
    for i in range(num_str-2):
        KEnew[i] = np.sum(weight * ((traj_diff[i]/2)**2).T) * 0.5 / (timestep**2 * conver1)
        TotE[i] =  KEnew[i] + potE_diff[i]
#    print("TotE\n",TotE,"\nKEnew\n",KEnew,"\npotE_diff\n",potE_diff)
    if np.prod((TotE - status["totE1stpoint"]) > totEdiff) > 0 or\
       np.prod((-TotE + status["totE1stpoint"]) > totEdiff) > 0:
        return -1
    return 0 

def force_calculation_run(origdir,scratchdir,runpoint=0):
    # The variable "runpoint" is a number to show runpointnumber of the trajectory calculation.
    # 1: 1st point, 2: 2ndpoint, 0: runpointnum is more than 3.
    ONIOM = int(progread.confread_each(origdir,"ONIOM",0)[1])
    prog          = progread.confread_each(origdir,"prog","gaussian")[1]
    highprog      = progread.confread_each(origdir,"highprog","gaussian")[1]
    diag = int(progread.confread_each(origdir,"diagnostics",0)[1])

    os.chdir(scratchdir)
    scfalgnum = 0 #scfalgnum reflects the forceoption1~4 you input on the top of this program
    status = progread.statusread(origdir,0,0)
    if status["progress"] == "g16Running": # change scfalgnum so as not to write trajectory into traj twice
        scfalgnum = 1
        if diag == 1:
            with open (origdir+"diagnostics",mode='a') as dw:
                dw.write(str(prog)+" calculation will be prepared\n")
                dw.flush()

    # Here start calculation of the whole input. If you are using ONIOM, real layer will be calculated.
    if prog == "gaussian":
        forcerun_result = g16run(programdir,origdir,scratchdir,"",scfalgnum,runpoint)
    elif prog == "mopac":
        forcerun_result = mopac2016run(programdir,origdir,scratchdir,"",scfalgnum,runpoint)
    if forcerun_result == -1:
        date = time.localtime()
        progread.errorlog(origdir,"SCF failure, choose appropriate scf algorithm"+\
                          time.strftime("%Y-%h-%d %H:%M:%S",date))
        aborttraj = int(progread.confread_each(origdir,"aborttraj",0)[1])
        if aborttraj == 0:
            if os.path.exits(scratchdir+outputlogfile) == True:
                shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
            else:
                progread.errorlog(origdir,"Output file was not found")
            sys.exit()
        elif aborttraj == 1:
            status = progread.statusread(origdir,"progress","g16Fail")
            if os.path.exists(scratchdir+outputlogfile) == True:
                isomernum = status["isomernum"]  
                shutil.copy(scratchdir+outputlogfile,origdir+"abortreason"+str(isomernum)+".log")
            return -1

    # Here start ONIOM calculation
    if ONIOM == 1:
        shutil.copyfile(scratchdir+outputlogfile,scratchdir+"R_"+outputlogfile)
        # First, make and calculate High layer.
        scfalgnum = 1
        if highprog == "gaussian":
            forcerun_result = g16run(programdir,origdir,scratchdir,"H_",scfalgnum,runpoint)
        elif highprog == "mopac":
            forcerun_result = mopac2016run(programdir,origdir,scratchdir,"H_",scfalgnum,runpoint)
        if forcerun_result == -1: 
            if os.path.exists(scratchdir+outputlogfile) == True:
                shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
            if os.path.exists(scratchdir+"R_"+outputlogfile) == True:
                shutil.copy(scratchdir+"R_"+outputlogfile,origdir+"R_"+outputlogfile)
            if os.path.exists(scratchdir+"H_"+outputlogfile) == True:
                shutil.copy(scratchdir+"H_"+outputlogfile,origdir+"H_"+outputlogfile)
            sys.exit()

        # Then calculate Low layer
        scfalgnum = 1
        if prog == "gaussian":
            forcerun_result = g16run(programdir,origdir,scratchdir,"L_",scfalgnum,runpoint)
        elif prog == "mopac":
            forcerun_result = mopac2016run(programdir,origdir,scratchdir,"L_",scfalgnum,runpoint)
        if forcerun_result == -1:
            aborttraj = int(progread.confread_each(origdir,"aborttraj",0)[1])
            if aborttraj == 0:
                if os.path.exists(scratchdir+outputlogfile) == True:
                    shutil.copy(scratchdir+outputlogfile,origdir+outputlogfile)
                if os.path.exists(scratchdir+"R_"+outputlogfile) == True:
                    shutil.copy(scratchdir+"R_"+outputlogfile,origdir+"R_"+outputlogfile)
                if os.path.exists(scratchdir+"H_"+outputlogfile) == True:
                    shutil.copy(scratchdir+"H_"+outputlogfile,origdir+"H_"+outputlogfile)
                if os.path.exists(scratchdir+"L_"+outputlogfile) == True:
                    shutil.copy(scratchdir+"L_"+outputlogfile,origdir+"L_"+outputlogfile)
                sys.exit()
            elif aborttraj == 1:
                status = progread.statusread(origdir,"progress","g16Fail")
                return -1

        # Calculate force using results obtained above and make g16.log file
        gausslog.oniomlogmake(origdir,scratchdir,scratchdir+"R_"+outputlogfile,scratchdir+"H_"+outputlogfile,\
                              scratchdir+"L_"+outputlogfile,scratchdir + outputlogfile)

    os.chdir(origdir)
    if runpoint == 0:
        status = progread.statusread(origdir,"progress","g16Done")

def g16run(programdir,origdir,scratchdir,prefix,scfalgnum,runpoint):
    status = progread.statusread(origdir,0,0)
    while True:
        if runpoint == 0:
            progcommake.progdynb(programdir,origdir,scratchdir,scfalgnum)
            if status["progress"] == "g16Prepared":
                status = progread.statusread(origdir,"progress","g16Running")
        if runpoint == 1:
            geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,\
            potentialE = progread.geoPlusVelread(origdir)
            progcommake.prog1stpoint(programdir,origdir,geoArr,velArr,atSym,atWeight,scfalgnum)
        if runpoint == 2:
            geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,\
            potentialE = progread.geoPlusVelread(origdir)
            progcommake.prog2ndpoint(programdir,origdir,scratchdir,geoArr,velArr,atSym,atWeight,\
                                     desiredModeEnK,KEinitmodes,KEinittotal,potentialE,scfalgnum)

        shutil.copy(origdir + prefix + inputcomfile, scratchdir + prefix + inputcomfile)
        subprocess.call(g16root+"g16 < "+scratchdir+prefix+inputcomfile \
                       +" > "+scratchdir+prefix+outputlogfile,shell=True)
        if proganal.donecheck(scratchdir+prefix+outputlogfile) == 1:
            # Fail-safe calculation implemented on 01/23/2019
            forcelimit = float(progread.confread_each(origdir,"forcelimit",1.0)[1])
            forceArr,newpotentialE = gausslog.forcereader(scratchdir+prefix+outputlogfile)
            if np.sum(np.abs(forceArr) > forcelimit) == 0:
                break
            else:
                if scfalgnum == 0:
                    scfalgnum = 2
                else:
                    scfalgnum += 1
                progread.errorlog(origdir,str(prefix)\
                                  +"g16 calc afforded very high force calculation, scfalg switched to "\
                                  +str(scfalgnum)+" runpointnum: "\
                                  +str(status["runpointnum"])+" isomernum: "+str(status["isomernum"]))
            ##
        else:
            if scfalgnum == 0:
                scfalgnum = 2
            else:
                scfalgnum += 1
            progread.errorlog(origdir,str(prefix)+"g16 calc failed, scfalg switched to "+str(scfalgnum)+" runpointnum: "\
                         +str(status["runpointnum"])+" isomernum: "+str(status["isomernum"]))
        if scfalgnum > 4:
            return -1
    return 0

def mopac2016run(programdir,origdir,scratchdir,prefix,scfalgnum,runpoint):
    diag = int(progread.confread_each(origdir,"diagnostics",0)[1])
    status = progread.statusread(origdir,0,0)
    while True:
        if runpoint == 0:
            progcommake.progdynb(programdir,origdir,scratchdir,scfalgnum)
            if status["progress"] == "g16Prepared":
                status = progread.statusread(origdir,"progress","g16Running")
        if runpoint == 1:
            if diag == 1:
                with open (origdir+"diagnostics",mode='a') as dw:
                    dw.write("let's make 1st point\n")
                    dw.flush()
            geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,\
            potentialE = progread.geoPlusVelread(origdir)
            progcommake.prog1stpoint(programdir,origdir,geoArr,velArr,atSym,atWeight,scfalgnum)
        if runpoint == 2:
            geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,\
            potentialE = progread.geoPlusVelread(origdir)
            progcommake.prog2ndpoint(programdir,origdir,scratchdir,geoArr,velArr,atSym,atWeight,\
                                     desiredModeEnK,KEinitmodes,KEinittotal,potentialE,scfalgnum)

        shutil.copy(origdir + prefix + inputcomfile, scratchdir + prefix + mopacfile +".mop")
        subprocess.call(mopacroot+mopacprogram+" "+scratchdir+prefix+ mopacfile +".mop",shell=True)
        if proganal.mopacdonecheck(scratchdir+prefix+mopacfile+".out") == 1:
            break
        else:
            if scfalgnum == 0:
                scfalgnum = 2
            else:
                scfalgnum += 1
            progread.errorlog(origdir,str(prefix)+"mopac calc failed, scfalg switched to "+str(scfalgnum)+" runpointnum: "\
                         +str(status["runpointnum"])+" isomernum: "+str(status["isomernum"]))
        if scfalgnum > 3:
            return -1
    gausslog.mopaclogmake(origdir,scratchdir,scratchdir+prefix+mopacfile+".out",scratchdir+prefix+outputlogfile)
        
    return 0


if __name__ == '__main__':
    # get origdir and scratchdir
    origdir = os.getcwd()
    if origdir[-1] != "/":
        origdir = origdir + "/"
    if os.path.exists(origdir+outputlogfile):
        os.remove(origdir+outputlogfile)
    args = sys.argv
    try:
        scratchdir = str(args[1])
        if os.path.exists(scratchdir) == False:
            print("specify scratchdir using absolute path")
            sys.exit()
    except:
        print("specify scratchdir")
        sys.exit()
    if scratchdir[-1] != "/":
        scratchdir = scratchdir + "/"
    status = progread.statusread(origdir,0,0)
    L1_firstloop(origdir,scratchdir)
    os.removedirs(scratchdir)
    sys.exit()


