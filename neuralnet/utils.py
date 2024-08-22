import os


def generate_unique_filename(directory, file_name, mode, extension):
    counter = 1
    file_path = os.path.join(directory, f"{file_name}_{mode}.{extension}")
    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{file_name}_{counter}_{mode}.{extension}")
        counter += 1
    return file_path
