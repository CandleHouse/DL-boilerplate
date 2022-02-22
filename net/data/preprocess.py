import os
import numpy as np


def get_file_list(folder, extension):
    file_list = []
    for dir_path, dir_names, file_names in os.walk(folder):
        for file in file_names:
            file_type = os.path.splitext(file)[1]
            if file_type == extension:
                file_fullname = os.path.join(dir_path, file)
                file_list.append(file_fullname)
    return file_list


def gen_input(folder, extension):
    images_files_list = np.sort(get_file_list(folder, extension))
    for i in range(len(images_files_list)):
        file = images_files_list[i]
        raw_data = np.fromfile(file, dtype=np.float32)


def gen_label(folder, extension):
    images_files_list = np.sort(get_file_list(folder, extension))


def gen_test(folder, extension):
    images_files_list = np.sort(get_file_list(folder, extension))


if __name__ == '__main__':
    gen_input()
    gen_label()
    gen_test()