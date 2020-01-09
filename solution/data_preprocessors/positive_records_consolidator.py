import os

'''
    Reads positive records from data folder and
    consolidates them into one file with
    decission attribute added

    TODO: finish...
'''

file_count = 0


def read_file(f):
    if f.mode == 'r':
        content = f.read()
        print(content)


for dirpath, dnames, fnames in os.walk("../../data/"):
    for f in fnames:
        abspath = os.path.join(dirpath, f)
        with open(abspath) as f:
            file_count += 1
            print("File number {}:".format(file_count))
            read_file(f)
