import os
import pandas as pd
import random
from random import randrange
from datetime import timedelta

def sort_AISgekoppeld(excel_file):
    info = pd.read_excel(excel_file)
    # print(info.head(4))
    info['Audio Start'] = pd.to_datetime(info['Audio Start'])
    info['Audio End'] = pd.to_datetime(info['Audio End'])
    return (
        info.assign(tmp=info["Audio End"] - info["Audio Start"])
        .sort_values(by="tmp", ascending=False)
        .drop(columns="tmp")
    )

class AISDataCuration():

    def __init__(self, AIS_folder, t):
        self.t = t
        self.Dstar = []
        self.AIS_folder = AIS_folder

    def _read_AIS_file(self, AIS_csv):
        AIS = pd.read_csv(AIS_csv)
        AIS['BaseDateTime'] = pd.to_datetime(AIS['BaseDateTime'])
        AIS = AIS.sort_values('BaseDateTime')
        AIS = AIS.reset_index(drop=True)
        return AIS

    def sum_by_key(self, dicts):
        result = {}

        for d in dicts:
            for k, v in d.items():
                result[k] = result.get(k, 0) + v

        return result

    def _entry_count(self):
        entry_count = []
        for file in os.listdir(self.AIS_folder):
            AIS_file = self._read_AIS_file(os.path.join(self.AIS_folder, file))
            summed_MMSI = AIS_file['MMSI'].value_counts().to_dict()
            entry_count.append(summed_MMSI)
        from collections import Counter
        c = Counter()
        for d in entry_count:
            c.update(d)
        return c

    def _get_list_of_AIS_files(self):
        list_dir = []
        for file in os.listdir(self.AIS_folder):
            list_dir.append(os.path.join(self.AIS_folder, file))
        return list_dir

    def _random_date(self, start, end):
        """
        This function will return a random datetime between two datetime
        objects.
        """
        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        try:
            random_second = randrange(int_delta)
            return start + timedelta(seconds=random_second)
        except:
            return start


    def _get_timeframe_AISrow(self, row, index, AIS):
        basetime = row['BaseDateTime']
        if index+1 < len(AIS) and AIS.iloc[[index+1]]['MMSI'].item() == AIS.iloc[[index]]['MMSI'].item():
            next_row_index = index + 1
            next_basetime = AIS.iloc[[next_row_index]]['BaseDateTime'].item()
            time_frame = self._random_date(basetime, next_basetime)
        elif AIS.iloc[[index-1]]['MMSI'].item() == AIS.iloc[[index]]['MMSI'].item():
            prev_row_index = index - 1
            prev_basetime = AIS.iloc[[prev_row_index]]['BaseDateTime'].item()
            time_frame = self._random_date(prev_basetime, basetime)
        else:
            time_frame = basetime
        return time_frame

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def subsample(self):
        D_star = {}
        entry_count = self._entry_count()
        tail_samples = [k for k, v in entry_count.items() if v < self.t]
        for t_k in tail_samples:
            entry_count[t_k] = self.t
        entry_prob_values = [self.t / v for _, v in entry_count.items()]
        entry_prob = dict(zip(entry_count.keys(), entry_prob_values))
        AISfiles = self._get_list_of_AIS_files()
        for file in AISfiles:
            list_times = []
            AIS = self._read_AIS_file(file)
            for index, row in AIS.iterrows():
                if random.random() < entry_prob[row['MMSI']]:
                    time_frame = self._get_timeframe_AISrow(row, index, AIS)
                    list_times.append(time_frame)
            D_star[file] = list_times
        print(len(self.flatten(list(D_star.values()))))
        return D_star

    def __call__(self):
        D_star = self.subsample()
        return D_star