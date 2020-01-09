import os

'''
    Reads positive records from data folder and
    consolidates them into one file with
    decission attribute added
    
'''

for dirpath, dnames, fnames in os.walk("../data/"):
    with open("../preprocessed_data/datafile.txt", "w") as datafile:
        for i in range(0, len(fnames), 2):
            datapoints_abspath = os.path.join(dirpath, fnames[i])
            targets_abspath = os.path.join(dirpath, fnames[i + 1])
            with open(datapoints_abspath) as datapoints:
                with open(targets_abspath) as targets:
                    targets_lines = targets.readlines()
                    for j, line in enumerate(datapoints):
                        if j != 0:
                            if targets_lines[j-1] == "1\n":
                                datafile.write(" ".join(line.split()[1:]) + " " + str(i//2) + "\n")
                            else:
                                datafile.write(" ".join(line.split()[1:]) + " -1\n")
