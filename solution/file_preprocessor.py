import glob


def write_lines(datapoints, targets_lines, datafile, i):
    for j, line in enumerate(datapoints):
        if j != 0:
            if targets_lines[j - 1] == "1\n":
                datafile.write(" ".join(line.split()[1:]) + " " + str(i) + "\n")
            else:
                datafile.write(" ".join(line.split()[1:]) + " -1\n")


def consolidate_files(src_dir="../data/", output_file="../preprocessed_data/datafile_v1.txt"):
    with open("../preprocessed_data/datafile_v1.txt", "w") as datafile:
        a_datapoints = glob.glob("../data/a*datapoints.txt")
        a_targets = glob.glob("../data/a*targets.txt")
        b_datapoints = glob.glob("../data/b*datapoints.txt")
        b_targets = glob.glob("../data/b*targets.txt")
        a_datapoints.sort()
        a_targets.sort()
        b_datapoints.sort()
        b_targets.sort()
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


def delete_unlabeled_records(src_file="../preprocessed_data/datafile_v1.txt",
                             output_file="../preprocessed_data/datafile_v2.txt"):
    with open(src_file) as datafile_v1:
        with open(output_file, "w") as datafile_v2:
            for line in datafile_v1:
                if line.split()[-1] != "-1":
                    datafile_v2.write(line)
