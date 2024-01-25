################################## bindgen.py ##################################
#
#    Make bind regulation file so as not to break solvent molecules during
#    high temperature annealing process
#    to run this program, type 
#    "python3 (path)/bindgen.py geometryfile(pdbformat) atom1 atom2 layer"
#    then exforceset parameter will be added to progdyn.conf automatically.
#
#    written by Hiroaki Kurouchi
#    ver 1.0 07/20/2018
#
################################################################################
import os
import sys
import linecache
import numpy as np
import progread

### Set parameter ###
k = 0.1
s = 1.5
forcetype = 2

def bindmake(filename,atom1,atom2,layer=999999):
    geoArr,molnum,layernum,atSym,atWeight = progread.pdbread(filename)
#    print(geoArr,molnum,layernum,atSym,atWeight)
    # This program make restriction only for molecules at the largest layer number if not specified
    firstatomnum = 0
    for i in range(len(layernum)):
        if layernum[i] == layer:
            firstatomnum = i
            break
    
    endatomnum = 0
    for i in range(firstatomnum,len(layernum)):
        endatomnum = i
        if layernum[i] != layer:
            break

#    print(firstatomnum,endatomnum)
#    print(layernum[firstatomnum],layernum[endatomnum])
    molatomnum = 0
    for i in range(firstatomnum,len(layernum)):
        molatomnum = i - firstatomnum
        if molnum[i] != molnum[firstatomnum]:
            break
   
    number_of_molecule = int((endatomnum - firstatomnum + 1) /molatomnum )

    length = np.linalg.norm(geoArr[firstatomnum+atom1-1]-geoArr[firstatomnum+atom2-1])    
    with open("progdyn.conf",mode="a") as bindwrite:
        bindwrite.write("##### bindrule section ####\n")
        for i in range(number_of_molecule):
            bindwrite.write("exforceset "+str(forcetype)+" "+\
                            str(atom1 + firstatomnum + i * molatomnum)+" "+\
                            str(atom2 + firstatomnum + i * molatomnum)+" "+\
                            str(k)+" "+"{: .2f}".format(s * length)+"\n")
    print("complete")

if __name__ == '__main__':
    filename = sys.argv[1]
    atom1 = int(sys.argv[2])
    atom2 = int(sys.argv[3])
    layer = int(sys.argv[4])
#    confreader("./")
    bindmake(filename,atom1,atom2,layer)



