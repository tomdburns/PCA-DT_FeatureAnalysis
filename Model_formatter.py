"""
Formats the csv file into a pkl read for model optimization
"""


import os
import numpy as np
import pandas as pd
import pickle as pkl


INFILE = 'BinaryMLFile_edit.csv'
OFILE  = 'BinaryMLFile_edit.pkl'

# ====================================================================
# DO NOT EDIT following lines
LOC = '\\'.join(os.path.realpath(__file__).split('\\')[:-1]) + '\\'
INFILE = LOC + INFILE
OFILE  = LOC + OFILE
# ====================================================================


OMIT = ['CAS']


def main():
    """Main execution of script"""
    data = pd.read_csv(INFILE)
    rcols, cols = [col for col in data], []
    for col in rcols:
        if col in OMIT:
            continue
        cols.append(col)
    frmtd = []
    for i in data.index:
        sub = []
        for col in cols:
            sub.append(data[col][i])
        frmtd.append(sub)
    frmtd = np.array(frmtd)
    pkl.dump(frmtd, open(OFILE, 'wb'))
    print(frmtd)


if __name__ in '__main__':
    main()
    print("\nProgram Terminated Normally.\n")
    input()
