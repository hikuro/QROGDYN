################################# progcommake.py ###############################
#
#    This code is based on prog1stpoint, prog2ndpoint, and progdynb
#    originally written by Daniel A. Singleton
#
#    written by Hiroaki Kurouchi
#    ver 06/16/2018 This does work
#    ver 06/21/2018 Sphereforce and ONIOM are implemented
#    ver 07/03/2018 Empiricalforce is implemented and some bugs are fixed
#    ver 07/04/2018 Soft-half harmonic potential is implemented
#    ver 08/09/2018 A numerical error is fixed. Never use ver 1.3 and before.
#    ver 09/05/2018 prog1stpoint and prog2ndpoint are revised for 
#                   restarting gaussian calculation after scf failure
#    ver 11/26/2018 Some calculation methods are changed
#    ver 04/17/2019 centeratomfix=2 was implemented
#    ver 04/26/2019 tailcheck was modified
#    ver 02/17/2020 damptimelimit was added       
#    ver 04/05/2020 modified for new style of gaussian output file
#    ver 04/12/2021 modified to specify layer numbers on centeratomfix option
#
################################################################################

import os
import copy
import linecache
import numpy as np
import sys
import numpy.matlib

import progdynstarterHP
import progread
import gausslog

#Define Constants
conver1=4.184E26 #dividing by this converts amu angs^2 /s^2 to kcal/mol
c=29979245800; h=6.626075E-34; avNum=6.0221415E23
RgasK=0.00198588; RgasJ=8.31447

def progdynb(programdir,origdir,scratchdir,scfalgnum=0):
    ### Initialize parameters ###
    status = progread.statusread(origdir,0,0)
    DRP           = int(progread.confread_each(origdir,"DRP",0)[1])
    isomernum     = status["isomernum"]
    runpointnum   = status["runpointnum"]

    # Equilibration-related parameters
    damping       = float(progread.confread_each(origdir,"damping",1)[1])
    tempcontrolperiod    = int(progread.confread_each(origdir,"tempcontrolperiod",0)[1])
    lightatom     = float(progread.confread_each(origdir,"lightatom",1)[1])
    autodamp      = int(progread.confread_each(origdir,"autodamp",0)[1])
    dampstarttime = float(progread.confread_each(origdir,"dampstarttime",999999)[1])
    #damptimelimit = float(progread.confread_each(origdir,"damptimelimit",9999)[1]) # This is directly added to tempwriteandcheck
    preequiv      = int(progread.confread_each(origdir,"preequiv",0)[1])
    preequivtime  = float(progread.confread_each(origdir,"preequivtime",0)[1])
    centeratomfix = int(progread.confread_each(origdir,"centeratomfix",0)[1])
    equivtemp     = float(progread.confread_each(origdir,"equivtemp",1000.0)[1])
    fixedatomnum  = int(progread.confread_each(origdir,"fixedatomnum",0)[1]) 

    # Parameters to control dynamics
    sphereon      = int(progread.confread_each(origdir,"sphereon",0)[1])
    timestep      = float(progread.confread_each(origdir,"timestep",1E-15)[1])
    empiricaldispersion = float(progread.confread_each(origdir,"empiricaldispersion",0)[1])
    exforce_endtime = int(progread.confread_each(origdir,"exforce_endtime",99999999)[1])
    exforce_num = int(progread.confread_count(origdir+progdynstarterHP.conffile,"exforceset"))
    fixedlayer = progread.confread_each(origdir+progdynstarterHP.conffile,"fixedlayer")[1:] 


    # Parameters for making comfile
    temperature   = float(progread.confread_each(origdir,"temperature",298.15)[1])
    charge        = int(progread.confread_each(origdir,"charge",0)[1])
    multiplicity  = int(progread.confread_each(origdir,"multiplicity",1)[1])
    method        = progread.confread_each(origdir,"method","HF/3-21G")[1]
    premethod     = progread.confread_each(origdir,"premethod","PM3")[1]
    pdbfile       = progdynstarterHP.pdbfile
    prog          = progread.confread_each(origdir,"prog","gaussian")[1]
    highprog      = progread.confread_each(origdir,"highprog","gaussian")[1]

    # read oniom parameters, highcharge and highmulti are values of high layer
    ONIOM         = int(progread.confread_each(origdir,"ONIOM",0)[1])
    # By now, ONIOM is not used for solvent geometry sampling, thus other factors are off.
    if ONIOM == 1:
        centeratomfix = 0
        autodamp = 0
    highcharge    = int(progread.confread_each(origdir,"highcharge",0)[1])
    highmulti     = int(progread.confread_each(origdir,"highmulti",0)[1])
    highatomnum   = int(progread.confread_each(origdir,"highatomnum",0)[1]) 
    highmethod    = progread.confread_each(origdir,"highmethod","HF/3-21G")[1]

    oldArr   = gausslog.inputstructurereader(origdir+"olddynrun")
    olderArr = gausslog.inputstructurereader(origdir+"olddynrun2")
    ## Fail-safe section for new style of Gaussian output file ##
    if np.sum(oldArr) == 0:
        oldArr   = gausslog.trajstructurereader(origdir+"traj",origdir+"olddynrun",1)
    if np.sum(olderArr) == 0:
        olderArr   = gausslog.trajstructurereader(origdir+"traj",origdir+"olddynrun",2)

    forceArr,newpotentialE = gausslog.forcereader(origdir+"olddynrun")
    olderforceArr,olderpotentialE = gausslog.forcereader(origdir+"olddynrun2")
    geoArr,velArr,atSym,weight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE\
    = progread.geoPlusVelread(origdir)

    # equilibration acceleration option
    if lightatom > 0 and runpointnum < preequivtime:
        weight = (weight > 10) * weight / lightatom + (weight <= 10) * weight 

    atomVel = np.linalg.norm(oldArr-olderArr,axis=1)/timestep
    KEold = np.sum(weight * (atomVel**2).T) * 0.5/conver1
    if centeratomfix > 0:
        if fixedatomnum == 0:
            KEold = np.sum(weight[highatomnum:] * (atomVel[highatomnum:]**2).T) * 0.5/conver1
        else:
            KEold = np.sum(weight[fixedatomnum:] * (atomVel[fixedatomnum:]**2).T) * 0.5/conver1


    # sanity check
    if not lightatom > 0 and runpointnum < preequivtime:
        if np.sum(atomVel > 1) > 0 :
            with open(origdir+"Errormessage",mode='a') as df:
                df.write("The loop is killed during sanity check")
                df.flush()
            sys.exit()

    apparentTemp = KEold * 2 / (3 * RgasK * len(atSym))
    if centeratomfix >0:
        apparentTemp = KEold * 2 / (3 * RgasK * len(atSym[highatomnum:]))
    newPotEK = (newpotentialE - potentialE) * 627.509
    # Add sphereforce
    if sphereon >= 1:
        forceArr,density,pressureAtm = applysphereforce(origdir,oldArr,weight,forceArr)

    # Add empiricaldispersion
    if empiricaldispersion > 0:
        forceArr,newPotEK = doempiricaldispersion(origdir,oldArr,atSym,forceArr,newPotEK)
    
    # Add extraforce
    if exforce_num > 0 and exforce_endtime > runpointnum:
        forceArr = doapplyexforce(origdir+progdynstarterHP.conffile,oldArr,forceArr)

########################### Then,start Verlet algorithm. ##############################
    newArr = copy.deepcopy(oldArr)
    Coeff = 1E20 * 627.509 * 4184 * 1000 / 0.529177
#    Coeff = 4.96147E29
    forceArr = (Coeff *  (timestep**2) * forceArr.T / weight).T
    originaldamping = copy.copy(damping)
    if autodamp == 0:
        damping = 1.0

    # Then we will calculate damping parameter and add up force and velocity
    # These are true only when sphereon > 0
    if sphereon > 0:
        if scfalgnum == 0:
            if autodamp == 1 and sphereon >= 1:
                tempwriteandcheck(origdir,apparentTemp,temperature,pressureAtm,density,runpointnum,killoption=1)
            elif autodamp > 1 and sphereon >= 1:
                # Shuffled tentatively
                if abs(apparentTemp - temperature) < 30:
                    damping = 1 + abs(1-originaldamping) * (temperature - apparentTemp) * 0.05
                elif temperature - apparentTemp >= 30:
                    damping = 1 / originaldamping
                tempwriteandcheck(origdir,apparentTemp,temperature,pressureAtm,density,runpointnum,killoption=autodamp)
            elif autodamp == 0 and tempcontrolperiod != 0:
                if  runpointnum % tempcontrolperiod == 0:
                    if apparentTemp > temperature:
                        damping = originaldamping 
                    elif apparentTemp > temperature:
                        damping = 1/originaldamping
                tempwriteandcheck(origdir,apparentTemp,temperature,pressureAtm,density,runpointnum,killoption=0)
            else:
                tempwriteandcheck(origdir,apparentTemp,temperature,pressureAtm,density,runpointnum,killoption=0)

        if autodamp > 0:
            if dampstarttime > runpointnum:
                if apparentTemp > equivtemp:
                    damping = originaldamping 
                elif apparentTemp - equivtemp <= -200:
                    damping = 1.05
                elif apparentTemp <= equivtemp:
                    damping = 1 / originaldamping


        # In any cases, the system avoids to get to 2000K.
        # If you want to simulate high temperature reaction, comment it out.
        if apparentTemp > 1800:
            damping = 0.95

        if os.path.exists(origdir + "thermoinfo") == True and scfalgnum == 0: 
            with open(origdir+"thermoinfo",mode='a') as th:
                th.write("  damping {: .5f}".format(damping))
                th.flush()

    if DRP != 1:
        newArr +=  damping * (oldArr-olderArr) + forceArr

    # Calculate the kinetic energy
    KEnew = np.sum(weight * (((newArr - olderArr)/2)**2).T) * 0.5 / (timestep**2 * conver1)
    Etotal = newPotEK + KEnew 
############################## End of Verlet algorithm #################################

    # Then fix the reactant atom if centeratomfix = 1
    if os.path.exists(origdir + pdbfile) == True and centeratomfix == 1:
        geoArrOrig,molnum,layernum,atSym,atWeight = progread.pdbread(origdir + pdbfile)
        alphabet_int = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,
                        "K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,
                        "T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}
        fixedlayer_modified  = []
        for info in fixedlayer:
            if info in [0,"0"]:
                fixedlayer_modified.append(0)
            elif type(info) is str:
                fixedlayer_modified.append(alphabet_int[info])

        tempArr = 0 * newArr.T
        for i in range(30):
            if i not in fixedlayer_modified:
                tempArr += (layernum == i) * newArr.T 
            else:
                tempArr += (layernum == i) * geoArrOrig.T
        
        newArr = tempArr.T

    # Or make a molecule which has zero-point vibrational energy
    if os.path.exists(origdir + pdbfile) == True and centeratomfix == 2:
        if fixedlayer != [0]:
            print("fixedlayer is not applicaple to centeratomfix >= 2")
        geoArrOrig,molnum,layernum,atSym,atWeight = progread.pdbread(origdir + pdbfile)
        amplitude,angular_velocity,phase,rotated_mode = progread.centeratomPhaseread(origdir,filename="centeratomPhase") 
        geoArrOrig = geoArrOrig[:highatomnum]
        geoArrZero = geoArrOrig + np.sum((amplitude * rotated_mode.T * np.sin(phase \
                      + runpointnum * angular_velocity)).T,axis=0)
        newArr[:highatomnum] =  geoArrZero

    ### print section
    if preequiv == 1 and runpointnum < preequivtime:
        method = premethod
    if prog == "gaussian":
        commake_standard(origdir,scfalgnum,newArr,atSym,charge,multiplicity,method,progdynstarterHP.inputcomfile)
    elif prog == "mopac":
        mopmake_standard(origdir,scfalgnum,newArr,atSym,charge,multiplicity,method,progdynstarterHP.inputcomfile)
    else:
        print("input proper program name")
        sys.exit()

    if ONIOM == 1:
        if highprog == "gaussian":
            commake_standard(origdir,scfalgnum,newArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,highmethod,"H_"+progdynstarterHP.inputcomfile)
        elif highprog == "mopac":
            mopmake_standard(origdir,scfalgnum,newArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,highmethod,"H_"+progdynstarterHP.inputcomfile)
        else:
            print("input proper program name")
            sys.exit()
        if prog == "gaussian":
            commake_standard(origdir,scfalgnum,newArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,method,"L_"+progdynstarterHP.inputcomfile)
        elif prog == "mopac":
            mopmake_standard(origdir,scfalgnum,newArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,method,"L_"+progdynstarterHP.inputcomfile)
        else:
            print("input proper program name")
            sys.exit()


    if scfalgnum == 0:
        putinfo2traj(origdir,newArr,atSym,newpotentialE,runpointnum,isomernum)
    

def prog1stpoint(programdir,origdir,geoArr,velArr,atSym,atWeight,scfalgnum=0):
    preequiv     = int(progread.confread_each(origdir,"preequiv",0)[1])
    premethod    = progread.confread_each(origdir,"premethod","PM3")[1]
    method       = progread.confread_each(origdir,"method","HF/3-21G")[1]
    highmethod   = progread.confread_each(origdir,"highmethod","HF/3-21G")[1]
    charge       = int(progread.confread_each(origdir,"charge",0)[1])
    multiplicity = int(progread.confread_each(origdir,"multiplicity",0)[1])
    ONIOM        = int(progread.confread_each(origdir,"ONIOM",0)[1])
    highcharge   = int(progread.confread_each(origdir,"highcharge",0)[1])
    highmulti    = int(progread.confread_each(origdir,"highmulti",0)[1])
    highatomnum  = int(progread.confread_each(origdir,"highatomnum",0)[1])
    prog         = progread.confread_each(origdir,"prog","gaussian")[1]
    highprog     = progread.confread_each(origdir,"highprog","gaussian")[1]

    if preequiv == 1:
        method = premethod
    if prog == "gaussian":
        commake_standard(origdir,scfalgnum,geoArr,atSym,charge,multiplicity,method,progdynstarterHP.inputcomfile)
    elif prog == "mopac":
        mopmake_standard(origdir,scfalgnum,geoArr,atSym,charge,multiplicity,method,progdynstarterHP.inputcomfile)
    else:
        print("input proper program name")
        sys.exit()

    if ONIOM == 1:
        if highprog == "gaussian":
            commake_standard(origdir,scfalgnum,geoArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,highmethod,"H_"+progdynstarterHP.inputcomfile)
        elif highprog == "mopac":
            mopmake_standard(origdir,scfalgnum,geoArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,highmethod,"H_"+progdynstarterHP.inputcomfile)
        else:
            print("input proper program name")
            sys.exit()
        if prog == "gaussian":
            commake_standard(origdir,scfalgnum,geoArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,method,"L_"+progdynstarterHP.inputcomfile)
        elif prog == "mopac":
            mopmake_standard(origdir,scfalgnum,geoArr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,method,"L_"+progdynstarterHP.inputcomfile)
        else:            
            print("input proper program name")
            sys.exit()   

def prog2ndpoint(programdir,origdir,scratchdir,geoArr,velArr,atSym,atWeight,\
                    desiredModeEnK,KEinitmodes,KEinittotal,potentialE,scfalgnum=0):
    preequiv      = int(progread.confread_each(origdir,"preequiv",0)[1])
    premethod     = progread.confread_each(origdir,"premethod","PM3")[1]
    method        = progread.confread_each(origdir,"method","HF/3-21G")[1]
    highmethod    = progread.confread_each(origdir,"highmethod","HF/3-21G")[1]
    charge        = int(progread.confread_each(origdir,"charge",0)[1])
    multiplicity  = int(progread.confread_each(origdir,"multiplicity",0)[1])
    ONIOM         = int(progread.confread_each(origdir,"ONIOM",0)[1])
    highcharge    = int(progread.confread_each(origdir,"highcharge",0)[1])
    highmulti     = int(progread.confread_each(origdir,"highmulti",0)[1])
    highatomnum   = int(progread.confread_each(origdir,"highatomnum",0)[1])
    prog         = progread.confread_each(origdir,"prog","gaussian")[1]
    highprog     = progread.confread_each(origdir,"highprog","gaussian")[1]

    status = progread.statusread(origdir,0,0)
    timestep     = float(progread.confread_each(origdir,"timestep",1E-15)[1])
    isomernum    = status["isomernum"]
    DRP          = int(progread.confread_each(origdir,"DRP",0)[1])
#    etolerance   = int(progread.confread_each(origdir,"etolerance",999)[1])
    forceArr1stpoint,potentialE1stpoint = gausslog.forcereader(origdir+"olddynrun2")
    if scfalgnum == 0:
        putinfo2traj(origdir,geoArr,atSym,potentialE1stpoint,1,isomernum)
    # The same procedure as addVelocities
    if not status["skipstart"] == "reverserestart":
        arr = geoArr + velArr
    elif status["skipstart"] == "reverserestart":  
        arr = geoArr - velArr
    forceArr,newpotentialE = gausslog.forcereader(scratchdir+progdynstarterHP.outputlogfile)   
    newPotentialEK = (newpotentialE - potentialE) * 627.509

#    # DRP section
#    if DRP == 0:
#        # Let's do Echeck
#        if status["skipstart"] == "forwardstart":
#            with open(origdir+"Echeck",mode='a') as ec:
#                ec.write("trajectory #"+str(isomernum)+"\n")
#                ec.write("point 1 potential E= "+ "{: .3f}".format(newPotentialEK)+\
#                         " point 1 kinetic E= "+"{: .3f}".format(KEinitmodes)+"  Total="+\
#                         "{: .3f}".format(newPotentialEK+KEinitmodes)+"\n")
#                ec.write("desired total energy= "+ "{: .3f}".format(desiredModeEnK)+"\n")
#                if (newPotentialEK + KEinitmodes) > (desiredModeEnK + etolerance) or\
#                   (newPotentialEK + KEinitmodes) < (desiredModeEnK - etolerance) :
#                    ec.write("XXXX bad total Energy \n")
#                ec.flush()

    # The same procedure as addForceEffect
    # originally, HalfCoeff = 0.5*1E23*627.509*4184*(timestep**2)/0.529177 
    HalfCoeff = 2.480736744E29
    forceArr = (HalfCoeff *  (timestep**2) * forceArr.T / atWeight).T
    if DRP == 1:
        forceArr *= 0
    arr = arr + forceArr

    # Then, make g16.com and input information into traj file
    if preequiv == 1:
        method = premethod
    if prog == "gaussian":
        commake_standard(origdir,scfalgnum,arr,atSym,charge,multiplicity,method,progdynstarterHP.inputcomfile)
    elif prog == "mopac":
        mopmake_standard(origdir,scfalgnum,arr,atSym,charge,multiplicity,method,progdynstarterHP.inputcomfile)
    else:
        print("input proper program name")
        sys.exit()

    if ONIOM == 1:
        if highprog == "gaussian":
            commake_standard(origdir,scfalgnum,arr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,highmethod,"H_"+progdynstarterHP.inputcomfile)
        elif highprog == "mopac":
            mopmake_standard(origdir,scfalgnum,arr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,highmethod,"H_"+progdynstarterHP.inputcomfile)
        else:
            print("input proper program name")
            sys.exit()
        if prog == "gaussian":
            commake_standard(origdir,scfalgnum,arr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,method,"L_"+progdynstarterHP.inputcomfile)
        elif prog == "mopac":
            mopmake_standard(origdir,scfalgnum,arr[:highatomnum],atSym[:highatomnum],\
                             highcharge,highmulti,method,"L_"+progdynstarterHP.inputcomfile)
        else:
            print("input proper program name")
            sys.exit()

    if scfalgnum == 0:
        putinfo2traj(origdir,arr,atSym,newpotentialE,2,isomernum)

def applysphereforce(origdir,oldArr,weight,forceArr):
    status = progread.statusread(origdir,0,0)
    runpointnum   = status["runpointnum"]
    sphereon      = int(progread.confread_each(origdir,"sphereon",0)[1])
    preequiv      = int(progread.confread_each(origdir,"preequiv",0)[1])
    preequivtime  = float(progread.confread_each(origdir,"preequivtime",0)[1])

    maxpressure   = float(progread.confread_each(origdir,"maxpressure",1000.0)[1])
    sphereforceK  = float(progread.confread_each(origdir,"sphereforceK",0.01)[1])    
    spheresize    = float(progread.confread_each(origdir,"spheresize",100)[1])
    splog         = int(progread.confread_each(origdir,"sphereforcelog",0)[1])

    distToOrig    = np.linalg.norm(oldArr,axis=1)

    if sphereon == 1:
        sphereforce   = (distToOrig > spheresize) * (distToOrig - spheresize) * sphereforceK 
        sphereforce   = (sphereforce > sphereforceK) * (sphereforceK - sphereforce) + sphereforce
    elif sphereon == 2: # soft half harmonic potential
        sphereforce   = (distToOrig > spheresize) * (distToOrig - spheresize)**2  * sphereforceK

    sphereforcetotal = np.sum(sphereforce)
    sphereforcetotalNewtons = sphereforcetotal * 627.509 * 4184 * 1E10 / (0.529177 * avNum)
    surfaceareaSqMeters = 4 * np.pi * spheresize**2 /1E20
    pressurePascal = sphereforcetotalNewtons / surfaceareaSqMeters
    pressureAtm   = pressurePascal / 101325

    # Then, define sphereforceK again
    if pressureAtm > maxpressure:
        sphereforceK *= maxpressure / pressureAtm

    if sphereon == 1:
        sphereforce   = (distToOrig > spheresize) * (distToOrig - spheresize) * sphereforceK
        sphereforce   = (sphereforce > sphereforceK) * (sphereforceK - sphereforce) + sphereforce
    elif sphereon == 2: # soft half harmonic potential
        sphereforce   = (distToOrig > spheresize) * (distToOrig - spheresize)**2  * sphereforceK

    if preequivtime > runpointnum:
        sphereforce  *= 100 

    sphereforcetotal = np.sum(sphereforce)
    forceArr += - (sphereforce * (oldArr.T / distToOrig) ).T
    if splog == 1:
        sphereforceX = - (sphereforce * (oldArr.T / distToOrig) ).T
        with open(origdir+"sphereforcelog",mode='a') as fw:
            fw.write("runpointnum "+str(runpointnum)+ " originalpressureAtm " + str(pressureAtm)+"\n")
            fw.write("sphereforceK "+str(sphereforceK)+"\n")
            for i in range(len(sphereforceX)):
                if np.linalg.norm(sphereforceX[i]) > 1E-6:
                    fw.write(str(i+1)+"  ")
                    for j in range(len(sphereforceX[i])):
                        fw.write(str(sphereforceX[i][j])+"  ")
                    fw.write("\n")
            fw.flush()

    # calculate the density at 0.9 * spheresize
    totalweight = np.sum((distToOrig < 0.9 * spheresize) * weight)
    density  = (totalweight / avNum) / (4/3 * np.pi * (0.9 * spheresize * 1E-8)**3)
    sphereforcetotalNewtons = sphereforcetotal * 627.509 * 4184 * 1E10 / (0.529177 * avNum)
    pressurePascal = sphereforcetotalNewtons / surfaceareaSqMeters
    pressureAtm   = pressurePascal / 101325
    if splog == 1:
        with open(origdir+"sphereforcelog",mode='a') as fw:
            fw.write("maxpressure "+str(maxpressure)+" newpressureAtm " + str(pressureAtm)+"\n")
            fw.flush()

    return forceArr,density,pressureAtm

def doempiricaldispersion(origdir,oldArr,atSym,forceArr,newPotEK):
    s6 = float(progread.confread_each(origdir,"empiricaldispersion",0)[1])
    # coeff[0] is c6 and coeff[1] is r0 in original progdynb program
    coeff = np.zeros([len(atSym),2]) # Nx2 matrix
    diffArrtensor = np.zeros([len(atSym),len(atSym),3]) # NxNx3 matrix
    for i in range(len(atSym)):
        for j in range(len(atSym)):
            diffArrtensor[i][j] = oldArr[i]
    # Here, diffArrtensor[i][j] is calcd to be vectors  between atom i and atom j (i -> j)
    diffArrtensor = -diffArrtensor + oldArr
    # distance between i and j is rij[i][j]
    rij = np.linalg.norm(diffArrtensor,axis=2) # NxN matrix
    # so as not to divide by zero, add one
    rij += np.identity(len(atSym))
    # direction is a tensor direction[i][j] shows direction vector from atom i to atom j
    direction = (diffArrtensor.T / rij.T).T
    for i in range(len(atSym)):
        if atSym[i]=="H":
            coeff[i] = [0.14,1.001]
#            coeff[i] = [0.16,1.11]
        elif atSym[i]=="He":
            coeff[i] = [0.08,1.012]
        elif atSym[i]=="Li":
            coeff[i] = [1.61,0.825]
        elif atSym[i]=="Be":
            coeff[i] = [1.61,1.408]
        elif atSym[i]=="B":
            coeff[i] = [3.13,1.485]
        elif atSym[i]=="C":
#            coeff[i] = [1.65,1.61]
            coeff[i] = [1.75,1.452]
        elif atSym[i]=="N":
            coeff[i] = [1.23,1.397]
        elif atSym[i]=="O":
            coeff[i] = [0.70,1.342]
        elif atSym[i]=="F":
            coeff[i] = [0.75,1.287]
        elif atSym[i]=="Ne":
            coeff[i] = [0.63,1.243]
        elif atSym[i]=="Na":
            coeff[i] = [5.71,1.144]
        elif atSym[i]=="Mg":
            coeff[i] = [5.71,1.364]
        elif atSym[i]=="Al":
            coeff[i] = [10.79,1.639]
        elif atSym[i]=="Si":
            coeff[i] = [9.23,1.716]
        elif atSym[i]=="P":
            coeff[i] = [7.84,1.705]
        elif atSym[i]=="S":
            coeff[i] = [5.57,1.683]
        elif atSym[i]=="Cl":
            coeff[i] = [5.07,1.639]
        elif atSym[i]=="Ar":
            coeff[i] = [4.61,1.595]
        elif atSym[i]=="K":
            coeff[i] = [10.8,1.485]
        elif atSym[i]=="Ca":
            coeff[i] = [10.8,1.474]
        elif atSym[i]=="Sc":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Ti":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="V":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Cr":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Mn":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Fe":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Co":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Ni":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Cu":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Zn":
            coeff[i] = [10.8,1.562]
        elif atSym[i]=="Ga":
            coeff[i] = [16.99,1.65]
        elif atSym[i]=="Ge":
            coeff[i] = [17.10,1.727]
        elif atSym[i]=="As":
            coeff[i] = [16.37,1.76]
        elif atSym[i]=="Se":
            coeff[i] = [12.64,1.771]
        elif atSym[i]=="Br":
            coeff[i] = [12.47,1.749]
        elif atSym[i]=="Pd":
            coeff[i] = [24.67,1.639]
        elif atSym[i]=="I":
            coeff[i] = [31.5,1.892]
    radiusmultiplier = float(progread.confread_each(origdir,"radiusmultiplier",1.25)[1])
    coeff[:,1] *= radiusmultiplier
    EdispH  = (1E6 / (4184 * 627.509)) * Edisp(s6,coeff,rij)
    # Subtract diagonal components which are artificially added to avoid division by Null
    EdispH  = EdispH * (1-np.identity(len(atSym))) 
    FdispHB = (0.52917725 * 1E6 / (4184 * 627.509)) * \
              (Edisp(s6,coeff,rij - 0.001) * (1-np.identity(len(atSym)))\
             - Edisp(s6,coeff,rij + 0.001) * (1-np.identity(len(atSym))) )/0.002
    EdispHtot = np.sum(EdispH)/2
    forceArrAll = - (FdispHB * direction.T).T
    forceArr +=  np.sum(forceArrAll,axis=1)
    newPotEK += EdispHtot
    return forceArr,newPotEK

def Edisp(s6,coeff,rij):
    cij = (coeff.T[0] * coeff[:,:1])**.5  # NxN matrix
    try:
        sumr0i = np.matlib.repmat(coeff.T[1],len(coeff.T[1]),1) # NxN matrix
    except:
        print("np.matlib doesn't work")
        sys.exit()
    sumr0ij = sumr0i + sumr0i.T # NxN matrix
    fdmp = 1 / (1 + np.exp(-20 * (rij / sumr0ij -1))) #NxN matrix
    Edisp = -s6 * cij * fdmp / rij**6
    return Edisp

def doapplyexforce(forcefile,oldArr,forceArr):
    exforce_num = int(progread.confread_count(forcefile,"exforceset"))
    exforceset = progread.confread_each_list_expanded(forcefile,"exforceset",5,exforce_num)
    for setnum in range(len(exforceset)):
        rij = np.linalg.norm(oldArr[int(exforceset[setnum][2])-1]-oldArr[int(exforceset[setnum][1])-1])
        # direction vector from i to j
        dij = (oldArr[int(exforceset[setnum][2])-1]-oldArr[int(exforceset[setnum][1])-1])/rij 
        if int(exforceset[setnum][0]) == 1:
            force = - float(exforceset[setnum][3]) * (rij - float(exforceset[setnum][4]))  
        if int(exforceset[setnum][0]) == 2:
            force = - float(exforceset[setnum][3]) * (rij - float(exforceset[setnum][4]))**2
        if int(exforceset[setnum][0]) == 3:
            force = - float(exforceset[setnum][3]) * (rij - float(exforceset[setnum][4]))**2  \
                    * ((rij - float(exforceset[setnum][4])) > 0 )
        forceArr[int(exforceset[setnum][1])-1] += -dij * force
        forceArr[int(exforceset[setnum][2])-1] += dij * force
    return forceArr

def tempwriteandcheck(origdir,apparentTemp,temperature,pressureAtm,density,runpointnum,killoption=0):
    print("apparentTemp",apparentTemp)
    with open(origdir+"thermoinfo",mode='a') as th:
        th.write("\nrunpointnum "+str(runpointnum)+"  temp {: .1f}".format(apparentTemp)+\
                 "   pressure {: .3f}".format(pressureAtm)+"  density {: .3f}".format(density))
        th.flush()
    linenum  = sum(1 for line in open(origdir+"thermoinfo"))
    if linenum > 100 and runpointnum > 300:
        temp = [] 
        for i in range(linenum - 100, linenum):
            Lineinfo = linecache.getline(origdir+"thermoinfo",i + 1).split()
            try:
                temp.append(float(Lineinfo[3]))
            except:
                pass
            temp = np.array(temp)
        if killoption == 1:
            if np.average(temp) <  temperature:
                with open(origdir+"thermoinfo",mode='a') as th:
                    th.write("\n temperature reached target temp, killoption=1")
                    th.flush()
                os.rename(origdir + "traj",origdir + "solvtraj")
                os.rename(origdir + "geoPlusVel",origdir + "solvgeoPlusVel")
                os.rename(origdir + "dynfollowfile",origdir + "solvdynfollowfile")
                os.rename(origdir + "thermoinfo", origdir + "solvthermoinfo")
                os.rename(origdir + "status", origdir + "solvstatus")
                sys.exit()
        elif killoption == 2:
            if abs(np.average(temp) - temperature) < 1 and np.std(temp) < 7:
                with open(origdir+"thermoinfo",mode='a') as th:
                    th.write("\n temperature reached target temp, killoption=2")
                    th.flush()
                os.rename(origdir + "traj",origdir + "solvtraj")
                os.rename(origdir + "geoPlusVel",origdir + "solvgeoPlusVel")
                os.rename(origdir + "dynfollowfile",origdir + "solvdynfollowfile")
                os.rename(origdir + "thermoinfo", origdir + "solvthermoinfo")
                os.rename(origdir + "status", origdir + "solvstatus")
                sys.exit()

        elif killoption == 3:
            allowedpressure = float(progread.confread_each(origdir,"allowedpressure",1000.0)[1])
            if abs(np.average(temp) - temperature) < 1 and np.std(temp) < 7 and\
                pressureAtm < allowedpressure:
                with open(origdir+"thermoinfo",mode='a') as th:
                    th.write("\n temperature reached target temp and pressure, killoption=3 averagedtemp="+\
                             str(np.average(temp))+"K")
                    th.flush()
                os.rename(origdir + "traj",origdir + "solvtraj")
                os.rename(origdir + "geoPlusVel",origdir + "solvgeoPlusVel")
                os.rename(origdir + "dynfollowfile",origdir + "solvdynfollowfile")
                os.rename(origdir + "thermoinfo", origdir + "solvthermoinfo")
                os.rename(origdir + "status", origdir + "solvstatus")
                sys.exit()

        damptimelimit = float(progread.confread_each(origdir,"damptimelimit",9999)[1])
        if runpointnum > damptimelimit:
            with open(origdir+"thermoinfo",mode='a') as th:
                th.write("\n trajectory did not reach target temp and pressure within timelimit, averagedtemp="+\
                 str(np.average(temp))+"K")
                th.flush()
                os.rename(origdir + "traj",origdir + "solvtraj")
                os.rename(origdir + "geoPlusVel",origdir + "solvgeoPlusVel")
                os.rename(origdir + "dynfollowfile",origdir + "solvdynfollowfile")
                os.rename(origdir + "thermoinfo", origdir + "solvthermoinfo")
                os.rename(origdir + "status", origdir + "solvstatus")
                sys.exit()
            
    linecache.clearcache()

def putinfo2traj(origdir,geoArr,atSym,potentialE,runpointnum,isomernum):
    title1,title2,title3,title4 = progread.confread_each(origdir,"title","you")[1],\
                              progread.confread_each(origdir,"title","need")[2],\
                              progread.confread_each(origdir,"title","a")[3],\
                              progread.confread_each(origdir,"title",progdynstarterHP.conffile)[4]
    with open(origdir+"traj",mode='a') as trajwrite:
        trajwrite.write(str(len(geoArr))+"\n")
        trajwrite.write(str(potentialE)+" "+title1+" "+title2+" "+title3+" "+title4+\
                        " runpoint "+ str(runpointnum)+" runisomer "+str(isomernum)+"\n")
        for i in range(len(geoArr)):
            trajwrite.write(" {:2}".format(atSym[i])+" "+"{: .8f} {: .8f} {: .8f} ".format(\
                            geoArr[i][0],geoArr[i][1],geoArr[i][2])+"\n")
        trajwrite.flush()


def commake_standard(origdir,scfalgnum,geoArr,atSym,charge,multiplicity,method,filename):
    status = progread.statusread(origdir,0,0)
    processors   = progread.confread_each(origdir,"processors",1)[1]
    memory       = progread.confread_each(origdir,"memory","1GB")[1]
    killcheck    = int(progread.confread_each(origdir,"killcheck",1)[1])
    checkpoint   = progread.confread_each(origdir,"checkpoint","g16.chk")[1]
    nonstandard  = int(progread.confread_each(origdir,"nonstandard",0)[1])
    method2      = progread.confread_each(origdir,"method2","")[1]
    method3      = progread.confread_each(origdir,"method3"," ")[1]
    method4      = progread.confread_each(origdir,"method4"," ")[1]
    method5      = progread.confread_each(origdir,"method5"," ")[1]
    method6      = progread.confread_each(origdir,"method6"," ")[1]
    title1,title2,title3,title4 = progread.confread_each(origdir,"title","you")[1],\
                                  progread.confread_each(origdir,"title","need")[2],\
                                  progread.confread_each(origdir,"title","a")[3],\
                                  progread.confread_each(origdir,"title",progdynstarterHP.conffile)[4]
    isomernum   = status["isomernum"]
    runpointnum = status["runpointnum"]
    highlevel   = int(progread.confread_each(origdir,"highlevel",999)[1])
    linkatoms   = int(progread.confread_each(origdir,"linkatoms",999)[1])
    with open(origdir + filename,mode="w") as comwrite:

    ### Route Section ###
        comwrite.write("%nproc="+str(processors)+"\n%mem="+str(memory)+"\n")
        if killcheck != 1:
            comwrite.write("%chk="+str(checkpoint)+"\n")
        if nonstandard == 0:
            ### Change SCF algorithm ###
            if scfalgnum == 0 or scfalgnum == 1:
                comwrite.write("#p "+str(method)+" "+str(progdynstarterHP.forceoption)+"\n")
            elif scfalgnum == 2:
                comwrite.write("#p "+str(method)+" "+str(progdynstarterHP.forceoption2)+"\n")
            elif scfalgnum == 3:
                comwrite.write("#p "+str(method)+" "+str(progdynstarterHP.forceoption3)+"\n")
            elif scfalgnum == 4:
                comwrite.write("#p "+str(method)+" "+str(progdynstarterHP.forceoption4)+"\n")
            elif scfalgnum == 5:
                comwrite.write("#p "+str(method)+" "+str(progdynstarterHP.forceoption5)+"\n")

            method2to4 = 0
            if len(method2) > 2:
                comwrite.write(str(method2)+" ")
                method2to4 = 1
            if len(method3) > 2:
                comwrite.write(str(method3)+" ")
                method2to4 = 1
            if len(method4) > 2:
                comwrite.write(str(method4)+" ")
                method2to4 = 1
            if method2to4 == 1:
                comwrite.write("\n")
        elif nonstandard == 1:
            comwrite.write("#p nonstd\n")
            if os.path.exists(origdir+"nonstandard") == True:
                linenum  = sum(1 for line in open(origdir+"nonstandard"))
                for i in range(linenum):
                    comwrite.write(linecache.getline(origdir+"nonstandard",i+1))
                linecache.clearcache()
                comwrite.write("\n")
        comwrite.write("\n") 

    ### Title section ###
        comwrite.write(title1+" "+title2+" "+title3+" "+title4+"\n")
        comwrite.write("runpoint  "+str(runpointnum)+"\n")
        comwrite.write("runisomer "+str(isomernum)+"\n\n") 

    ### Geometry section ###
        comwrite.write(str(charge)+" "+str(multiplicity)+"\n")
        for i in range(len(geoArr)):
            comwrite.write(" {:2}".format(atSym[i])+" "+"{: .7f} {: .7f} {: .7f} ".format(\
                     geoArr[i][0],geoArr[i][1],geoArr[i][2]))
            comwrite.write("\n")
        comwrite.write("\n")

    ### Add option ###
        if len(method5) > 2:
            comwrite.write(str(method5)+"\n")
        if len(method6) > 2:
            comwrite.write(str(method6)+"\n")
        if os.path.exists(origdir+"methodfile") == True:
            linenum  = sum(1 for line in open(origdir+"methodfile"))
            for i in range(linenum):
                comwrite.write(linecache.getline(origdir+"methodfile",i+1))
                linecache.clearcache()
            comwrite.write("\n")
        comwrite.write("\n")
        comwrite.flush()

def mopmake_standard(origdir,scfalgnum,geoArr,atSym,charge,multiplicity,method,filename):
    status = progread.statusread(origdir,0,0)
    processors   = progread.confread_each(origdir,"processors",1)[1]
#    memory       = progread.confread_each(origdir,"memory",2000000)[1]
    title1,title2,title3,title4 = progread.confread_each(origdir,"title","you")[1],\
                                  progread.confread_each(origdir,"title","need")[2],\
                                  progread.confread_each(origdir,"title","a")[3],\
                                  progread.confread_each(origdir,"title",progdynstarterHP.conffile)[4]
    isomernum   = status["isomernum"]
    runpointnum = status["runpointnum"]
#    highlevel   = int(progread.confread_each(origdir,"highlevel",999)[1])
#    linkatoms   = int(progread.confread_each(origdir,"linkatoms",999)[1])
    with open(origdir + filename,mode="w") as comwrite:

    ### Route Section ###
        ### Change SCF algorithm ###
        if scfalgnum == 0 or scfalgnum == 1:
            comwrite.write(str(method)+" "+str(progdynstarterHP.mopacforceoption))
        elif scfalgnum == 2:
            comwrite.write(str(method)+" "+str(progdynstarterHP.mopacforceoption2))
        elif scfalgnum == 3:
            comwrite.write(str(method)+" "+str(progdynstarterHP.mopacforceoption3))
        comwrite.write(" threads="+str(processors)+" charge="+str(charge))
        if multiplicity == 2:
            comwrite.write(" DOUBLET")
        elif multiplicity == 3:
            comwrite.write(" TRIPLET")
        comwrite.write(" \n")

    ### Title section ###
        comwrite.write(title1+" "+title2+" "+title3+" "+title4+"\n")
        comwrite.write("runpoint  "+str(runpointnum))
        comwrite.write("  runisomer "+str(isomernum)+"\n") 

    ### Geometry section ###
        for i in range(len(geoArr)):
            comwrite.write(" {:2}".format(atSym[i])+" "+"{: .7f} {: .7f} {: .7f} ".format(\
                     geoArr[i][0],geoArr[i][1],geoArr[i][2]))
            comwrite.write("\n")
        comwrite.write("\n")
        comwrite.flush()



if __name__ == "__main__":
    geoArr,velArr,atSym,atWeight,desiredModeEnK,KEinitmodes,KEinittotal,potentialE\
    = progread.geoPlusVelread("./")
#    prog1stpoint("./","./",geoArr,velArr,atSym,atWeight)
#    prog2ndpoint("./","./","./scratchdir/",geoArr,velArr,atSym,atWeight,\
#    desiredModeEnK,KEinitmodes,KEinittotal,potentialE)
    forceArr = np.zeros([len(atSym),3])
    forceArr = doempiricaldispersion("./",geoArr,atSym,forceArr,0)



