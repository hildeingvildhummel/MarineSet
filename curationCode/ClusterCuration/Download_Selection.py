from datetime import datetime
import numpy as np
from GoogleCloudConnection import get_file_names, download_file
import soundfile as sf
import os

def find_two_closest(target, date_list):
    differences = []
    # Compute absolute differences
    for value in date_list:
        try:
            differences.append(np.abs(value - target))
        except Exception as e:
            print('excepting: ', str(e))
            continue
    differences = np.array(differences)

    # Get indices of the two smallest differences
    closest_indices = np.argsort(differences)[:2]

    # Retrieve the two closest datetime values
    closest_dates = np.array(date_list)[closest_indices.astype(int)]

    return closest_dates, closest_indices

def read_audio_section(filename, start_time, stop_time):
    track = sf.SoundFile(filename)

    can_seek = track.seekable() # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    start_frame = sr * start_time
    frames_to_read = sr * (stop_time - start_time)
    track.seek(start_frame)
    audio_section = track.read(frames_to_read)
    return audio_section, sr

def extract_time_frame_from_audio(time, file):
    try:
        start_file = np.datetime64(datetime.strptime(' '.join(file.split('.')[0].split('_')[-2:]), "%Y%m%d %H%M%S"), 's')
    except:
        start_file = np.datetime64(datetime.strptime(' '.join(file.split('.')[0].split('_')[-1:]), "%y%m%d%H%M%S"),
                                   's')
    start_secs = (np.datetime64(time, 's') - start_file).astype(int)
    audio, sr = read_audio_section(file, start_secs, start_secs + 10)
    return audio, sr
def load_values_incrementally(input_pickle):
    """
    Incrementally load the values from a pickle file, returning the values
    by their index across all keys, without loading the full dictionary into memory.
    """
    import pickle
    with open(input_pickle, 'rb') as f:
        # Load the dictionary incrementally (in chunks)
        while True:
            try:
                obj = pickle.load(f)  # Load one object at a time (part of the dictionary)

                if isinstance(obj, dict):  # Ensure it's a dictionary
                    # Convert all the values in the dictionary to lists
                    values_list = list(obj.values())

                    # Determine the maximum number of values in any list (key)
                    max_len = max(len(val) for val in values_list)
                    print('MAXIMUM: ', max_len)

                    # Iterate over the indices of the values in the lists
                    for i in range(max_len):
                        # Collect values from all keys at index i
                        yield [val[i] for val in values_list if i < len(val)]  # Yield as a list
            except EOFError:
                break  # End of the pickle file

def search_substrings_in_filenames(directory, substring, substring2):
    for root, _, files in os.walk(directory):
        for file in files:
            if substring in file and substring2 in file:
                return True
    return False

def sampled_curation(pickle_file, curation_save_path, temp_save_path):
    prev_path = None
    counter = 0
    for value_group in load_values_incrementally(pickle_file):
        path = value_group[2]
        result = path.split('/audio/', 1)
        path = result[0].rsplit('/', 1)[1] + '/audio/' + result[1]
        hydrophone = path.split('/')[-2]
        time_selected = value_group[1]
        time_selected = np.datetime64(time_selected)
        string_time = np.datetime_as_string(time_selected,'s').replace('T','_').replace(":", "").replace("-", "")
        if search_substrings_in_filenames(curation_save_path, string_time, hydrophone.upper()):
            counter += 1
            continue
        print('counter: ', counter)
        if prev_path == path:
            file_list = prev_file_list
            start_times = prev_start_times
        else:
            file_list = get_file_names("noaa-passive-bioacoustic", path)
            start_times = []
            for x in file_list:
                if 'Post-Deployment' in x:
                    continue
                try:
                    start_time = np.datetime64(datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:]), "%Y%m%d %H%M%S"),'s')
                except Exception as e:
                    if ' '.join(x.split('.')[0].split('_')[-2:]).endswith('60'):
                        start_time = np.datetime64(datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:])[:-2] + "59", "%Y%m%d %H%M%S"))
                    elif len(' '.join(x.split('.')[0].split('_')[-1:])) > 6:
                        start_time = np.datetime64(datetime.strptime(' '.join(x.split('.')[0].split('_')[-1:]), "%y%m%d%H%M%S"),'s')
                    elif ' '.join(x.split('.')[0].split('_')[-2:]).endswith('o'):
                        start_time = np.datetime64(
                            datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:])[:-2], "%y%m%d%H%M%S"), 's')
                    else:
                        try:
                            start_time = np.datetime64(
                                datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:]), "%y%m%d %H%M%S"), 's')
                        except:
                            print('No start times: ', str(e), ' '.join(x.split('.')[0].split('_')[-2:]))
                            counter += 1
                            continue
                start_times.append(start_time)
        value, index = find_two_closest(time_selected, start_times)
        for time, indices in zip(value, index):
            if time_selected - time >= 0:
                file = file_list[indices]
                break
        download_file("noaa-passive-bioacoustic", file,
                      os.path.join(temp_save_path, '{}'.format(file.split('/')[-1])))
        try:
            audio, sr = extract_time_frame_from_audio(time_selected, os.path.join(temp_save_path, '{}'.format(
                file.split('/')[-1])))
            sf.write(os.path.join(curation_save_path, '{}_{}.flac'.format(file.split('/')[-1].split('.')[0],
                                                                          np.datetime_as_string(time_selected,
                                                                                                's').replace('T',
                                                                                                             '_').replace(
                                                                              ":", "").replace("-", ""))), audio, sr)
        except Exception as e:
            print(str(e))
        os.remove(os.path.join(temp_save_path, '{}'.format(file.split('/')[-1])))

        counter += 1
        prev_file_list = file_list
        prev_start_times = start_times
    return audio

