import os
import shutil
import pandas as pd
import datetime

def check_dir(MYDIR):
    # Check if the directory exists
    CHECK_FOLDER = os.path.isdir(MYDIR)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)


def FairTrainTestSplitDeepship(Deepship_path):
    classes = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    trade_off = datetime.datetime.strptime('20171201', '%Y%m%d')
    for class_level in classes:
        sub_path = os.path.join(Deepship_path, class_level)
        files = os.listdir(sub_path)
        check_dir(os.path.join(os.path.join(os.path.join(Deepship_path, 'Fair'), 'Training'), class_level))
        check_dir(os.path.join(os.path.join(os.path.join(Deepship_path, 'Fair'), 'Test'), class_level))
        for file in files:
            date_string = file[:8]
            datestamp = datetime.datetime.strptime(date_string, '%Y%m%d')
            if datestamp < trade_off:
                print(os.path.join(sub_path, file), os.path.join(
                    os.path.join(os.path.join(os.path.join(Deepship_path, 'Fair'), 'Training'), class_level), file))
                shutil.copyfile(os.path.join(sub_path, file), os.path.join(
                    os.path.join(os.path.join(os.path.join(Deepship_path, 'Fair'), 'Training'), class_level), file))
            else:
                print(os.path.join(sub_path, file), os.path.join(
                    os.path.join(os.path.join(os.path.join(Deepship_path, 'Fair'), 'Test'), class_level), file))
                shutil.copyfile(os.path.join(sub_path, file), os.path.join(
                    os.path.join(os.path.join(os.path.join(Deepship_path, 'Fair'), 'Test'), class_level), file))


def rearrange_wav_files_shipsEar(shipsEar_path):
    """This function rearranges the data types of the extracted ShipsEar data. It moves
    the data with different extensions to the corresponding subfolder. If the subfolder
    does not exist, it will create the subfolder.
    Output:
        rearranged files"""
    # List the files in the selected directory
    files = os.listdir(shipsEar_path)
    # Define the extensions of the data
    dataset_list = ['train', 'val', 'test']
    # Iterate over the files
    for split in dataset_list:
        # Define the directory of interest
        MYDIR = shipsEar_path + split
        print(MYDIR)
        check_dir(MYDIR)
        file = 'shipsEar_{}.csv'.format(split)
        split_df = pd.read_csv(shipsEar_path + file)
        print(file, shipsEar_path + file)
        file_series = split_df['']
        # Define the file path
        MYFILE = shipsEar_path + file
        # Check if the file exists
        CHECK_FILE = os.path.isfile(MYFILE)
        print(CHECK_FILE)
        # If so, move to the correct subfolder.
        if CHECK_FILE:
            shutil.move(shipsEar_path + file, shipsEar_path + split + '/' + file)