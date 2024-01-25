#################################### gausslog.py #######################################
#
# ver 07/15/2017
# ver 06/11/2018 revised for QuasiPROGDYN
# ver 06/21/2018 oniomlogmake is implemented
# ver 09/07/2018 revised for MOLPRO2016 
#
# written for python 3.6
# This program reads gaussian09 output file and returns calculated parameters
# MOLPRO output files are also read and edited.
#
# written by Hiroaki Kurouchi
#
#########################################################################################

import sys
import math
import numpy as np
import linecache
import time
import progread
import os
import copy

# define parameters
number_atom  = {
   '1':'H','2':'He','3':'Li','4':'Be','5':'B',
   '6':'C','7':'N','8':'O','9':'F','10':'Ne',
   '11':'Na','12':'Mg','13':'Al','14':'Si','15':'P',
   '16':'S','17':'Cl','18':'Ar','19':'K','20':'Ca',
   '21':'Sc','22':'Ti','23':'V','24':'Cr','25':'Mn',
   '26':'Fe','27':'Co','28':'Ni','29':'Cu','30':'Zn',
   '31':'Ga','32':'Ge','33':'As','34':'Se','35':'Br',
   '36':'Kr','37':'Rb','38':'Sr','39':'Y','40':'Zr',
   '41':'Nb','42':'Mo','43':'Tc','44':'Ru','45':'Rh',
   '46':'Pd','47':'Ag','48':'Cd','49':'In','50':'Sn',
   '51':'Sb','52':'Te','53':'I','54':'Xe','55':'Cs','56':'Ba'}

atom_weight = {'H':1.00784,'He':4.0026,'Li':6.941,'Be':9.012,'B':10.811,
               'C':12.0,'N':14.007,'O':15.9994,'F':18.9984,'Ne':20.1797,
               'Na':22.989,'Mg':24.305,'Al':26.98154,'Si':28.0855,'P':30.9738,
               'S':32.066,'Cl':35.4527,'Ar':39.948,'K':39.0983,'Ca':40.078,
               'Sc':44.96,'Ti':47.867,'V':50.94,'Cr':51.9961,'Mn':54.938,
               'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,
               'Ga':69.723,'Ge':72.64,'As':74.9216,'Se':78.96,'Br':79.904,
               'Pd':106.42,'I':126.90447}


def titlereader(filename,title1):
    linenumtot  = sum(1 for line in open(filename))
    linenum = 1
    title = "Error for some reason :-<"
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) == 0:
            lineinfo = "empty"
        if lineinfo[0] == title1:
            title = lineinfo
            break
        linenum += 1
    linecache.clearcache()
    return title

def structurereader(filename):
    # this function reads standard orientation of the molecule.
    # the last structure in the logfile is read
    Natoms      = atomnum(filename)
    linenumtot  = sum(1 for line in open(filename))
    linenum = 1
    flagnum = 0
    linenum = 1
    thermoflag = 0
    structure  = np.zeros([Natoms,3])
    AtomNumber = [6 for i in range(Natoms)]
    Atoms      = [i for i in range(Natoms)]
    Atomweight = np.zeros([Natoms])
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) <= 1:
            lineinfo = ["empty" for i in range(10)]
        if lineinfo[0] == "Standard" and lineinfo[1] == "orientation:":
            flagnum = 1
        if lineinfo[0] == "Rotational" :
            flagnum = 0
        try:
            if int(lineinfo[0]) > 0 and flagnum == 1:
                AtomNumber[int(lineinfo[0])-1]       = int(lineinfo[1])
                for i in range(3):
                    structure[int(lineinfo[0])-1][i] = float(lineinfo[i+3])
        except:
            pass
        if lineinfo[1] == "Thermochemistry":
            thermoflag = 1
        if lineinfo[0] == "Molecular":
            thermoflag = 0
        try:
            if lineinfo[0] == "Atom" and thermoflag == 1:
                Atomweight[int(lineinfo[1])-1] = float(lineinfo[8])
        except:
            pass
        linenum += 1
    for atm in range(Natoms):
        Atoms[atm] = number_atom[str(AtomNumber[atm])]
    linecache.clearcache()
    return structure,Atoms,Atomweight

def inputstructurereader(filename):
    # this function reads standard orientation of the molecule.
    # the last structure in the logfile is read
    linenumtot  = sum(1 for line in open(filename))
    Natoms      = atomnum(filename)
    flagnum = 0
    linenum = 1
    structure  = np.zeros([Natoms,3])
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) <= 1:
            lineinfo = ["empty" for i in range(10)]
        if lineinfo[0] == "Input" and lineinfo[1] == "orientation:":
            flagnum = 1
        if lineinfo[0] == "Distance" or lineinfo[0] == "Standard":
            flagnum = 0
        try:
            if int(lineinfo[0]) > 0 and flagnum == 1:
                for i in range(3):
                    structure[int(lineinfo[0])-1][i] = float(lineinfo[i+3])
        except:
            pass
        linenum += 1

    linecache.clearcache()
    return structure

def trajstructurereader(filename,filename2,number):
    Natoms      = atomnum(filename2)
    structure  = np.zeros([Natoms,3])
    linenumtot  = sum(1 for line in open(filename))
    for i in range(Natoms):
        lineinfo = linecache.getline(filename,linenumtot-number*(Natoms+2)+3+i).split()
        for j in range(3):
            structure[i][j] = float(lineinfo[j+1])
    linecache.clearcache()
    return structure

def getinputstructures(filename,strnum):
    Natoms      = atomnum(filename)
    inputstructures = []
    linenum = 0
    for i in range(strnum):
        inputstr,linenum = inputstructurereader_multiple(filename,linenum)
        inputstructures.append(inputstr)
    return inputstructures

def inputstructurereader_multiple(filename,startline):
    # this function reads standard orientation of the molecule.
    # the last structure in the logfile is read
    linenumtot  = sum(1 for line in open(filename))
    Natoms      = atomnum(filename)
    flagnum = 0
    linenum = startline
    structure  = np.zeros([Natoms,3])
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if flagnum > 0 and len(lineinfo) == 1 and "----" in lineinfo[0]:
           flagnum += 1 
        if len(lineinfo) <= 1:
            lineinfo = ["empty" for i in range(10)]
        if lineinfo[0] == "Input" and lineinfo[1] == "orientation:":
#            print("input orientation found, linenum = ",linenum)
            flagnum = 1
#        if flagnum == 1:
#            print(lineinfo)
#        if flagnum > 0 and ("------" in lineinfo[0]):
#            print("----- found, flagnum = ",flagnum)
#            flagnum += 1
        try:
            if int(lineinfo[0]) > 0 and flagnum > 1:
                for i in range(3):
                    structure[int(lineinfo[0])-1][i] = float(lineinfo[i+3])
        except:
            pass
        if flagnum == 4:
            break
        linenum += 1

    linecache.clearcache()
    return structure,linenum


def mopacinputstructurereader(filename):
    # this function reads standard orientation of the molecule.
    # the last structure in the logfile is read
    linenumtot  = sum(1 for line in open(filename))
    Natoms      = atomnum_mopac(filename)
    flagnum = 0
    linenum = 1
    structure  = np.zeros([Natoms,3])
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) <= 1:
            lineinfo = ["empty" for i in range(10)]
        if lineinfo[0] == "ATOM" and lineinfo[1] == "CHEMICAL":
            flagnum = 1
        if lineinfo[0] == "CARTESIAN" or lineinfo[0] == "COORDINATES":
            flagnum = 0
            break
        try:
            if int(lineinfo[0]) > 0 and flagnum == 1:
                for i in range(3):
                    structure[int(lineinfo[0])-1][i] = float(lineinfo[2*i+2])
        except:
            pass
        linenum += 1
    linecache.clearcache()

    return structure

def grrmstructurereader(filename):
    # this function reads standard orientation of the molecule.
    # the last structure in the logfile is read
    linenumtot  = sum(1 for line in open(filename))
    Natoms      = atomnum_grrm(filename)
    flagnum = 0
    linenum = 1
    atoms = ["null" for i in range(Natoms)]
    structure  = np.zeros([Natoms,3])
    atom_number = 0
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) <= 1:
            lineinfo = ["empty" for i in range(10)]
        if lineinfo[0] == "empty" and flagnum == 1:
            break
        if len(lineinfo) == 4 and flagnum == 1:
            atoms[atom_number] = lineinfo[0]
            for i in range(3):
                structure[atom_number][i] = float(lineinfo[i+1])
            atom_number += 1
        if lineinfo[0] == "Geometry" and lineinfo[1] == "(Origin":
            flagnum = 1
        linenum += 1
    linecache.clearcache()

    return structure,atoms


def forcereader(filename):
    Natoms      = atomnum(filename)
    linenumtot  = sum(1 for line in open(filename))
    forceflag = 0
    potentialE = 0
    linenum = 1
    force  = np.zeros([Natoms,3])
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) <= 3:
            lineinfo = ["empty" for i in range(10)]
        if lineinfo[0] == "Energy=" and lineinfo[2] == "NIter=":
            potentialE = float(lineinfo[1])
        if lineinfo[0] == "SCF"  and lineinfo[1] == "Done:":
            potentialE = float(lineinfo[4])
        if lineinfo[0] == "ONIOM:"  and lineinfo[1] == "extrapolated":
            potentialE = float(lineinfo[4])
        if lineinfo[0] == "E2"  and lineinfo[3] == "EUMP2":
            potentialE = float(lineinfo[5][:-4])*1000
        # tempstring is not implemented
        if lineinfo[0] == "Total":
            try:
                if float(lineinfo[4]) < 0:
                    potentialE = float(lineinfo[4])
            except:
                pass
                
        if lineinfo[0] == "Center" and lineinfo[1] == "Atomic" and lineinfo[2] == "Forces":
            forceflag = 1
        try:
            if int(lineinfo[0]) > 0 and forceflag == 1:
                for i in range(3):
                    force[int(lineinfo[0])-1][i] = float(lineinfo[i+2])
        except:
            pass

        if lineinfo[1] == "Forces:":
            forceflag = 0

        linenum += 1
    linecache.clearcache()
    return force,potentialE 

def mopacforcereader(filename):
    Natoms = atomnum_mopac(filename)
    linenumtot  = sum(1 for line in open(filename))
    forceflag = 0
    potentialE = 0
    force  = np.zeros([Natoms,3])
    linenum = 1
    cart = {"X":0,"Y":1,"Z":2}
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) <= 3:
            lineinfo = ["empty" for i in range(10)]
        try:
            if lineinfo[7] == "KCAL/ANGSTROM":
                force[int(lineinfo[1])-1][cart[lineinfo[4]]] = - float(lineinfo[6]) * 0.52917725 / 627.509
        except:
            pass
        if lineinfo[0] == "FINAL" and lineinfo[1] == "HEAT":
            potentialE = float(lineinfo[5]) / 627.509
        linenum += 1

    return force,potentialE

def normal_mode_reader(filename):
    # This module should not be used for linear atoms
    # get number of atoms
    flagnum     = 1
    Natoms      = atomnum(filename) 
    linenumtot  = sum(1 for line in open(filename))
    
    flagfreq = 0
    linenum  = 1 
    Normal_Modes     = np.zeros([3*Natoms-6,Natoms,3])
    Frequencies     = np.zeros([3*Natoms-6])
    Force_Constants = np.zeros([3*Natoms-6]) 
    Reduced_Masses  = np.zeros([3*Natoms-6])
    readnum         = 1
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) == 0:
            lineinfo = "empty"
        if lineinfo[0] == "Harmonic" and lineinfo[1] == "frequencies":
            flagfreq += 1
        try:
            if int(lineinfo[0]) == readnum + 5 and flagfreq == 1:
                readnum        += 5
                readnormalmodes = 0
        except:
            pass
        if lineinfo[0] == "Frequencies" and lineinfo[1] == "---":
            for i in range(5):
                try:
                    Frequencies[readnum+i-1] = float(lineinfo[i+2]) 
                except:
                    pass
        if lineinfo[0] == "Reduced" and lineinfo[2] == "---":
            for i in range(5):
                try:
                    Reduced_Masses[readnum+i-1] = float(lineinfo[i+3])
                except:
                    pass
        if lineinfo[0] == "Force" and lineinfo[2] == "---":
            for i in range(5):
                try:
                    Force_Constants[readnum+i-1] = float(lineinfo[i+3])
                except:
                    pass
        if lineinfo[0] == "Coord" and lineinfo[1] == "Atom":
            readnormalmodes = 1
        try:
            if readnormalmodes == 1 and int(lineinfo[0]) < 4:
                for i in range(5):
                    try:
                        Normal_Modes[readnum+i-1][int(lineinfo[1])-1][int(lineinfo[0])-1] =\
                         float(lineinfo[i+3])  
                    except:
                        pass
        except:
            pass
        if flagfreq == 2:
            break
        linenum += 1
    linecache.clearcache()
#    print(Normal_Modes,Frequencies,Force_Constants,Reduced_Masses)
    return Normal_Modes,Frequencies,Force_Constants,Reduced_Masses  

def grrm_freqcalcreader(filename):
    # This module should not be used for linear atoms
    # get number of atoms
    flagnum     = 1
    Natoms      = atomnum_grrm(filename)
    linenumtot  = sum(1 for line in open(filename))
    geoArr,atSym = grrmstructurereader(filename)
    atWeight = np.zeros(Natoms)
    for atnum in range(Natoms):
        atWeight[atnum] = float(atom_weight[atSym[atnum]])

    linenum  = 1
    mode     = np.zeros([3*Natoms-6,Natoms,3])
    freq     = np.zeros([3*Natoms-6])
    force    = np.zeros([3*Natoms-6])
    redMass  = np.zeros([3*Natoms-6])
    reading_modes   = 0
    atom_number     = 0
    readmodeflag    = 0
    coordinate_dict = {"x":0,"y":1,"z":2}
    redmassreadflag = 0
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) < 4:
            lineinfo = ["empty" for i in range(4)]
        if lineinfo[0] == "Freq." and lineinfo[1] == ":":
            reading_modes += 3
            if reading_modes  > 3*Natoms-6:
                break
            for i in range(3):
                freq[reading_modes-3 +i] = float(lineinfo[i+2])
        if lineinfo[0] == "Red." and lineinfo[1] == "M":
            for i in range(3):
                redMass[reading_modes-3 +i] = float(lineinfo[i+3])
            readmodeflag = 1
            redmassreadflag = 1
            lineinfo[3] = ":"
        if readmodeflag == 1 and (lineinfo[2] in ["x","y","z"]):
            xyz = coordinate_dict[lineinfo[2]]
            for i in range(3):
                try:
                    mode[reading_modes-3 +i][atom_number][xyz] = float(lineinfo[i+4])
                except:
                    print("Non")
                    pass
            if xyz == 2:
                atom_number += 1
        if lineinfo[3] != ":":
            atom_number = 0
            readmodeflag = 0
        else:
            if redmassreadflag == 1:
                atom_number = 0
                redmassreadflag = 0
        linenum += 1

    linecache.clearcache()
    ### force = (2pi)**2 * 10E-2 (=mDyn to kg/s**2) * (2.99792458E10 * freq)**2 * (1.6605402E-27 * redMass
    force = 5.891834E-7 * freq**2 * redMass

    return geoArr,atSym,atWeight,mode,freq,force,redMass 

def zpereader(filename):
    linenum     = 1
    linenumtot  = sum(1 for line in open(filename))
    zpeGauss = 0
    zpePlusE  = 0
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) < 4:
            lineinfo = [ 0 for i in range(4)]
        try:
            if lineinfo[0] == "Zero-point":
                zpeGauss = float(lineinfo[2])
            elif lineinfo[4] == "zero-point":
                zpePlusE  = float(lineinfo[6])
        except:
            pass
        linenum += 1
    linecache.clearcache()
    return  zpeGauss, zpePlusE

def atomnum(filename):
    Natoms = 0
    linenum = 1 
    linenumtot  = sum(1 for line in open(filename))
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) == 0:
            lineinfo = "empty"
        if lineinfo[0] == "NAtoms=":
            Natoms = int(lineinfo[1])
            break
        linenum += 1

    linenum = 1
    if Natoms == 0:
        flag = 0
        while linenum < linenumtot:
            lineinfo = linecache.getline(filename,linenum).split()
            if len(lineinfo) == 0:
                lineinfo = "empty"
            if lineinfo[0] == "Center":
                flag = 1
            try:
                if int(lineinfo[0]) > Natoms and flag == 1:
                    Natoms = int(lineinfo[0])
            except:
                pass
            try:
                if int(lineinfo[0]) < Natoms and flag == 1:
                    break
            except:
                pass
            linenum += 1
    #print(Natoms)        
    return Natoms

def atomnum_mopac(filename):
    Natoms = 0
    linenum = 1
    linenumtot  = sum(1 for line in open(filename))
    if Natoms == 0:
        flag = 0
        while linenum < linenumtot:
            lineinfo = linecache.getline(filename,linenum).split()
            if len(lineinfo) == 0:
                lineinfo = "empty"
            if lineinfo[0] == "ATOM":
                flag = 1
            try:
                if int(lineinfo[0]) > Natoms and flag == 1:
                    Natoms = int(lineinfo[0])
            except:
                pass
            try:
                if int(lineinfo[0]) < Natoms and flag == 1:
                    break
            except:
                pass
            linenum += 1
    #print(Natoms)        
    return Natoms

def atomnum_grrm(filename):
    Natoms = 0
    linenum = 1
    linenumtot  = sum(1 for line in open(filename))
    flag = 0
    while linenum < linenumtot:
        lineinfo = linecache.getline(filename,linenum).split()
        if len(lineinfo) == 0:
            lineinfo = ["empty"]
        if lineinfo[0] == "ENERGY":
            break
        if len(lineinfo) == 4 and flag == 1:
            Natoms += 1
        if lineinfo[0] == "INPUT" and lineinfo[1] == "ORIENTATION":
            flag = 1
        linenum += 1

    linecache.clearcache()
    #print(Natoms)        
    return Natoms

def mopaclogmake(origdir,scratchdir,mopaclog,output):
    geoArr = mopacinputstructurereader(mopaclog)
    status = progread.statusread(origdir,0,0)
    force,potE = mopacforcereader(mopaclog)
    title1,title2,title3,title4 = progread.confread_each(origdir,"title","you")[1],\
                          progread.confread_each(origdir,"title","need")[2],\
                          progread.confread_each(origdir,"title","a")[3],\
                          progread.confread_each(origdir,"title","title")[4]
    title = [title1,title2,title3,title4,"runpoint ",str(status["runpointnum"]),"isomernum ",str(status["isomernum"])]
    date    = time.localtime()
    with open(output,mode="w") as log:
        log.write("This file was created by mopaclogmake \n")

        ### Title section ###
        log.write("---------------------------------------------------------------------\n--TITLE--\n")
        for num in range(len(title)):
            log.write(title[num]+" ")
        log.write("\n")
        log.write("---------------------------------------------------------------------\n")

        ### Input Geometry section ###
        log.write("NAtoms= "+str(len(geoArr))+"\n")
        log.write("                         Input orientation:\n")
        log.write("---------------------------------------------------------------------\n")
        log.write(" Center     Atomic      Atomic             Coordinates (Angstroms)\n")
        log.write(" Number     Number       Type             X           Y           Z\n")
        log.write("---------------------------------------------------------------------\n")
        for i in range(len(geoArr)):
            log.write("  {:>5}  ".format(i+1)+"     1       "+"     1       "+\
            "{: 7f}".format(geoArr[i][0])+"  "+
            "  "+"{: 7f}".format(geoArr[i][1])+"  "+"{: 7f}".format(geoArr[i][2])+"\n")
        log.write("---------------------------------------------------------------------\n")
        log.write("\n\nDistance matrix (angstroms): \n")

        log.write("---------------------------------------------------------------------\n")
        log.write("                         Standard  orientation:\n")
        log.write("---------------------------------------------------------------------\n")
        log.write(" Center     Atomic      Atomic             Coordinates (Angstroms)\n")
        log.write(" Number     Number       Type             X           Y           Z\n")
        log.write("---------------------------------------------------------------------\n")
        for i in range(len(geoArr)):
            log.write("  {:>5}  ".format(i+1)+"     1       "+"     0       "+"{: 7f}".format(geoArr[i][0])+"  "+
                   "  "+"{: 7f}".format(geoArr[i][1])+"  "+"{: 7f}".format(geoArr[i][2])+"\n")
        log.write("---------------------------------------------------------------------\n")
        log.write("Rotational \n\n")
        log.write("SCF Done:  E(MOPAC) =  " +"{: 8f}".format(potE)+"    A.U. \n\n")
        log.write("-------------------------------------------------------------------\n")
        log.write(" Center     Atomic                   Forces (Hartrees/Bohr)\n")
        log.write(" Number     Number              X              Y              Z\n")
        log.write("-------------------------------------------------------------------\n")
        for i in range(len(geoArr)):
            log.write("  {:>5}  ".format(i+1)+"     1           "+"{: 7f}".format(force[i][0])+"  "+
                   "  "+"{: 7f}".format(force[i][1])+"  "+"{: 7f}".format(force[i][2])+"\n")
        log.write("-------------------------------------------------------------------\n")
        log.write("Cartesian Forces:  Max     Unknown  RMS     Unknown\n\n")
        log.write("Normal termination of Gaussian Program "+time.strftime("%Y-%h-%d %H:%M:%S",date))

    return 0

def oniomlogmake(origdir,scratchdir,reallog,highlog,lowlog,output):
    realArr = inputstructurereader(reallog)      # we need realArr as coordinate of the whole system
    realforce,realpotE = forcereader(reallog)
    highforce,highpotE = forcereader(highlog)
    lowforce,lowpotE   = forcereader(lowlog) 
    force = copy.deepcopy(realforce) 

    force[:len(highforce)] = force[:len(highforce)] + highforce - lowforce
    potE = realpotE + highpotE - lowpotE

    title1  = progread.confread_each(origdir,"title","you")[1]
    title   = titlereader(reallog,title1)
    date    = time.localtime()

    with open(output,mode="w") as log:
        log.write("This file was created by oniomlogmake \n")

        ### Title section ###
        log.write("---------------------------------------------------------------------\n--TITLE--\n")
        for num in range(len(title)):
            log.write(title[num]+" ")
        log.write("\n")
        log.write("---------------------------------------------------------------------\n")

        ### Input Geometry section ###
        log.write("NAtoms= "+str(len(realArr))+"\n")
        log.write("                         Input orientation:\n")
        log.write("---------------------------------------------------------------------\n")
        log.write(" Center     Atomic      Atomic             Coordinates (Angstroms)\n")
        log.write(" Number     Number       Type             X           Y           Z\n")
        log.write("---------------------------------------------------------------------\n")
        for i in range(len(realArr)):
            if i < len(highforce):
                log.write("  {:>5}  ".format(i+1)+"     1       "+"     0       "+\
                "{: 7f}".format(realArr[i][0])+"  "+
                "  "+"{: 7f}".format(realArr[i][1])+"  "+"{: 7f}".format(realArr[i][2])+"\n")
            else:
                log.write("  {:>5}  ".format(i+1)+"     1       "+"     1       "+\
                "{: 7f}".format(realArr[i][0])+"  "+
                "  "+"{: 7f}".format(realArr[i][1])+"  "+"{: 7f}".format(realArr[i][2])+"\n")
        log.write("---------------------------------------------------------------------\n")
        log.write("\n\nDistance matrix (angstroms): \n")

        log.write("---------------------------------------------------------------------\n")
        log.write("                         Standard  orientation:\n")
        log.write("---------------------------------------------------------------------\n")
        log.write(" Center     Atomic      Atomic             Coordinates (Angstroms)\n")
        log.write(" Number     Number       Type             X           Y           Z\n")
        log.write("---------------------------------------------------------------------\n")
        for i in range(len(realArr)):
            log.write("  {:>5}  ".format(i+1)+"     1       "+"     0       "+"{: 7f}".format(realArr[i][0])+"  "+
                   "  "+"{: 7f}".format(realArr[i][1])+"  "+"{: 7f}".format(realArr[i][2])+"\n")
        log.write("---------------------------------------------------------------------\n")
        log.write("Rotational \n\n")

        log.write("-------------------------------------------------------------------\n")
        log.write(" Center     Atomic                RealForces (Hartrees/Bohr)\n")
        log.write(" Number     Number              X              Y              Z\n")
        log.write("-------------------------------------------------------------------\n")
        for i in range(len(realforce)):
            log.write("  {:>5}  ".format(i+1)+"     1           "+"   {: 7f}".format(realforce[i][0])+"  "+
                   "  "+"{: 7f}".format(realforce[i][1])+"  "+"{: 7f}".format(realforce[i][2])+"\n")
        log.write("-------------------------------------------------------------------\n\n\n")
        log.write("-------------------------------------------------------------------\n")
        log.write(" Center     Atomic               High_Layer_Forces (Hartrees/Bohr)\n")
        log.write(" Number     Number              X              Y              Z\n")
        log.write("-------------------------------------------------------------------\n")
        for i in range(len(highforce)):
            log.write("  {:>5}  ".format(i+1)+"     1           "+"{: 7f}".format(highforce[i][0])+"  "+
                   "  "+"{: 7f}".format(highforce[i][1])+"  "+"{: 7f}".format(highforce[i][2])+"\n")
        log.write("-------------------------------------------------------------------\n\n\n")
        log.write("-------------------------------------------------------------------\n")
        log.write(" Center     Atomic                Low_Layer_Forces (Hartrees/Bohr)\n")
        log.write(" Number     Number              X              Y              Z\n")
        log.write("-------------------------------------------------------------------\n")
        for i in range(len(lowforce)):
            log.write("  {:>5}  ".format(i+1)+"     1           "+"{: 7f}".format(lowforce[i][0])+"  "+
                   "  "+"{: 7f}".format(lowforce[i][1])+"  "+"{: 7f}".format(lowforce[i][2])+"\n")
        log.write("-------------------------------------------------------------------\n\n\n")


        log.write("Potential Energy of Real Layer= " + "{: 8f}".format(realpotE) + "\n")
        log.write("Potential Energy of High Layer= " + "{: 8f}".format(highpotE) + "\n")
        log.write("Potential Energy of Low Layer= " + "{: 8f}".format(lowpotE) + "\n")
        log.write("SCF Done:  E(ONIOM) =  " +"{: 8f}".format(potE)+"    A.U. \n\n")
        log.write("-------------------------------------------------------------------\n")
        log.write(" Center     Atomic                   Forces (Hartrees/Bohr)\n")
        log.write(" Number     Number              X              Y              Z\n")
        log.write("-------------------------------------------------------------------\n")
        for i in range(len(realArr)):
            log.write("  {:>5}  ".format(i+1)+"     1           "+"{: 7f}".format(force[i][0])+"  "+
                   "  "+"{: 7f}".format(force[i][1])+"  "+"{: 7f}".format(force[i][2])+"\n")
        log.write("-------------------------------------------------------------------\n")
        log.write("Cartesian Forces:  Max     Unknown  RMS     Unknown\n\n")
        log.write("Normal termination of Gaussian Program "+time.strftime("%Y-%h-%d %H:%M:%S",date))
    


if __name__ == "__main__":
    filename = sys.argv[1]
#    structure,Atoms,Atomweight = structurereader(filename)    
#    print(structure,Atoms)

    inputstructures = getinputstructures(filename,15)
    print(inputstructures)

#    freqfile = sys.argv[1]
#    geoArr,atSym,atWeight,mode,freq,force,redMass = grrm_freqcalcreader(freqfile)
#    geoArr,atSym,atWeight = structurereader(freqfile)
#    mode,freq,force,redMass = normal_mode_reader(freqfile)
#    zpeGauss, zpePlusE = zpereader(freqfile)
#    print("geoArr", geoArr,"atSym",atSym,"atWeight",atWeight)

#    print("mode",mode,"freq",freq,"force",force,"redMass",redMass)
#    print("zpeGauss",zpeGauss,"zpePlusE",zpePlusE)

#    Normal_Modes,Frequencies,Force_Constants,Reduced_Masses = normal_mode_reader(filename)
#    print("structure",structure,"Atoms",Atoms,"Atomweight",Atomweight,
#          "Normal mode", Normal_Modes, "Frequencies",Frequencies,
#           "Force Constant",Force_Constants, "Red Mass",Reduced_Masses)
#    origdir = os.getcwd()
#    if origdir[-1] != "/":
#        origdir = origdir + "/" 
#    scratchdir = sys.argv[1]
#    oniomlogmake(origdir,scratchdir,scratchdir+"R_g16.log",scratchdir+"H_g16.log",\
#                 scratchdir+"L_g16.log",scratchdir+"temp_g16.log")
    #print(force)
#    force,potentialE = mopacforcereader(filename)
#    structure = mopacinputstructurereader(filename)
#    print(structure,force,potentialE)

