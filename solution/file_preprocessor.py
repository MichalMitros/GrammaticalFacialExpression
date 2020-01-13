import os


def consolidate_files(src_dir="../data/", output_file="../preprocessed_data/datafile_v1.txt"):
    n = 0
    for dirpath, dnames, fnames in os.walk(src_dir):
        with open(output_file, "w") as datafile:
            for f in range(0, len(fnames)):
                nameparts = fnames[f].split("_")
                if nameparts[len(nameparts) - 1] == "datapoints.txt":
                    filename = ""
                    for i, part in enumerate(nameparts):
                        if i != len(nameparts) - 1:
                            filename += part
                            filename += "_"
                    datapoints_abspath = os.path.join(dirpath, filename + "datapoints.txt")
                    targets_abspath = os.path.join(dirpath, filename + "targets.txt")
                    with open(datapoints_abspath) as datapoints:
                        with open(targets_abspath) as targets:
                            targets_lines = targets.readlines()
                            for j, line in enumerate(datapoints):
                                if j != 0:
                                    if targets_lines[j - 1] == "1\n":
                                        datafile.write(" ".join(line.split()[1:]) + " " + str(n) + "\n")
                                    else:
                                        datafile.write(" ".join(line.split()[1:]) + " -1\n")
                    n = n + 1

def delete_unlabeled_records(src_file="../preprocessed_data/datafile_v1.txt", output_file="../preprocessed_data/datafile_v2.txt"):
    with open(src_file) as datafile_v1:
        with open(output_file, "w") as datafile_v2:
            for line in datafile_v1:
                if line.split()[-1] != "-1":
                    datafile_v2.write(line)
