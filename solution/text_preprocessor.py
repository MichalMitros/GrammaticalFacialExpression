import glob

'''
    Reads positive records from data folder and
    consolidates them into one file with
    decision attribute added and unlabeled records deleted

'''


def write_lines(datapoints, targets_lines, datafile, i):
    for j, line in enumerate(datapoints):
        if j != 0:
            if targets_lines[j - 1] == "1\n":
                datafile.write(" ".join(line.split()[1:]) + " " + str(i) + "\n")
            else:
                datafile.write(" ".join(line.split()[1:]) + " -1\n")


with open("../preprocessed_data/datafile_v1.txt", "w") as datafile:
    a_datapoints = glob.glob("../data/a*datapoints.txt")
    a_targets = glob.glob("../data/a*targets.txt")
    b_datapoints = glob.glob("../data/b*datapoints.txt")
    b_targets = glob.glob("../data/b*targets.txt")
    for i in range(len(a_datapoints)):
        with open(a_datapoints[i]) as datapoints:
            with open(a_targets[i]) as targets:
                targets_lines = targets.readlines()
                write_lines(datapoints, targets_lines, datafile, i)
    for i in range(len(b_datapoints)):
        with open(b_datapoints[i]) as datapoints:
            with open(b_targets[i]) as targets:
                targets_lines = targets.readlines()
                write_lines(datapoints, targets_lines, datafile, i)

'''
    Deleting unlabeled records
'''
with open("../preprocessed_data/datafile_v1.txt") as datafile_v1:
    with open("../preprocessed_data/datafile_v2.txt", "w") as datafile_v2:
        for line in datafile_v1:
            if line.split()[-1] != "-1":
                datafile_v2.write(line)
