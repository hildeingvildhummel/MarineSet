import os
import pickle
import numpy as np
from datetime import datetime
import soundfile as sf
from tqdm import tqdm

from AISdata import sort_AISgekoppeld
from GoogleCloudConnection import get_file_names, download_file

class AISAudioExtractor:
    """
    Self-contained AIS → audio extractor.
    No dependency on RandomSelection anymore.
    """

    def __init__(
        self,
        info_excel,
        curated_pickle,
        bucket_name="noaa-passive-bioacoustic",
        temp_dir="temp",
        output_dir="AIS_output",
        min_date="2019-01-01",
        clip_seconds=10,
    ):
        self.bucket = bucket_name
        self.info_pd = sort_AISgekoppeld(info_excel)
        self.curated = self._load_curated(curated_pickle)

        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.min_date = np.datetime64(min_date)
        self.clip_seconds = clip_seconds

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    # ==============================================================
    # Loading
    # ==============================================================

    def _load_curated(self, pickle_path):
        with open(pickle_path, "rb") as f:
            curated = pickle.load(f)
        return {k: v for k, v in curated.items() if v}

    @staticmethod
    def _find_two_closest(target, date_list):
        """Return two closest datetimes + indices."""
        differences = [np.abs(value - target) for value in date_list]
        differences = np.array(differences)

        closest_indices = np.argsort(differences)[:2]
        closest_dates = np.array(date_list)[closest_indices]

        return closest_dates, closest_indices

    @staticmethod
    def _read_audio_section(filename, start_time, stop_time):
        """Read part of audio file using seek (fast)."""
        track = sf.SoundFile(filename)

        if not track.seekable():
            raise ValueError("File not seekable")

        sr = track.samplerate
        start_frame = int(sr * start_time)
        frames_to_read = int(sr * (stop_time - start_time))

        track.seek(start_frame)
        audio_section = track.read(frames_to_read)

        return audio_section, sr

    def _extract_time_frame_from_audio(self, timestamp, filepath):
        """Extract clip around AIS timestamp."""
        start_file = np.datetime64(
            datetime.strptime(
                " ".join(filepath.split(".")[0].split("_")[-2:]),
                "%Y%m%d %H%M%S",
            ),
            "s",
        )

        start_secs = (np.datetime64(timestamp, "s") - start_file).astype(int)

        return self._read_audio_section(
            filepath,
            start_secs,
            start_secs + self.clip_seconds,
        )

    # ==============================================================
    # Helpers
    # ==============================================================

    def _get_cloud_path(self, ais_file):
        data_file = ais_file.split("/")[-1]
        row = self.info_pd.loc[self.info_pd["AIS data file"] == data_file]
        if row.empty:
            print(f"⚠️ Warning: AIS file '{data_file}' not found in info Excel. Skipping.")
            return None
        return row.iloc[:, 0:3].agg("/".join, axis=1).values[0]

    @staticmethod
    def _parse_start_times(file_list):
        start_times = []
        for x in file_list:
            if 'Post-Deployment' in x:
                continue
            print(x)
            # print(' '.join(x.split('.')[0].split('_')[-2:]))
            try:
                start_time = np.datetime64(
                    datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:]), "%Y%m%d %H%M%S"), 's')
            except Exception as e:
                if ' '.join(x.split('.')[0].split('_')[-2:]).endswith('60'):
                    start_time = np.datetime64(
                        datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:])[:-2] + "59", "%Y%m%d %H%M%S"))
                elif len(' '.join(x.split('.')[0].split('_')[-1:])) > 6:
                    print(x)
                    start_time = np.datetime64(
                        datetime.strptime(' '.join(x.split('.')[0].split('_')[-1:]), "%y%m%d%H%M%S"), 's')
                elif ' '.join(x.split('.')[0].split('_')[-2:]).endswith('o'):
                    # print('Example print: ', ' '.join(x.split('.')[0].split('_')[-2:])[:-2], ' '.join(x.split('.')[0].split('_')[-2:]))
                    start_time = np.datetime64(
                        datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:])[:-2], "%y%m%d%H%M%S"), 's')
                else:
                    try:
                        start_time = np.datetime64(
                            datetime.strptime(' '.join(x.split('.')[0].split('_')[-2:]), "%y%m%d %H%M%S"), 's')
                    except:
                        print('No start times: ', str(e), ' '.join(x.split('.')[0].split('_')[-2:]))
                        continue
            start_times.append(start_time)
        # return [
        #     np.datetime64(
        #         datetime.strptime(
        #             (lambda s: s[:-2] + f"{int(s[-2:]) - 1:02d}" if s[-2:] == "60" else s)(" ".join(x.split(".")[0].split("_")[-2:])),
        #             "%Y%m%d %H%M%S",
        #         ),
        #         "s",
        #     )
        #     for x in file_list
        # ]
        return start_times

    def _choose_audio_file(self, timestamp, file_list, start_times):
        values, indices = self._find_two_closest(timestamp, start_times)

        for time, idx in zip(values, indices):
            if timestamp - time > 0:
                return file_list[idx]

        return None

    # ==============================================================
    # Core
    # ==============================================================

    def build_file_dict(self):
        file_dict = {}

        for ais_file, timestamps in tqdm(
            self.curated.items(),
            desc="Matching AIS → audio",
            unit="AIS",
        ):
            path = self._get_cloud_path(ais_file)

            file_list = get_file_names(self.bucket, path)
            start_times = self._parse_start_times(file_list)

            for ts in timestamps:
                ts = np.datetime64(ts)
                if ts < self.min_date:
                    continue

                google_file = self._choose_audio_file(ts, file_list, start_times)
                if google_file is None:
                    continue

                file_dict.setdefault(google_file, []).append(ts)

        return file_dict

    def process(self):
        file_dict = self.build_file_dict()

        for selected_file, timestamps in tqdm(
            file_dict.items(),
            desc="Processing audio files",
            unit="file",
        ):
            local_path = os.path.join(
                self.temp_dir,
                selected_file.split("/")[-1],
            )

            download_file(self.bucket, selected_file, local_path)

            for ts in tqdm(
                timestamps,
                leave=False,
                desc="Extracting clips",
                unit="clip",
            ):
                try:
                    audio, sr = self._extract_time_frame_from_audio(ts, local_path)

                    out_name = (
                        f"{selected_file.split('/')[-1].split('.')[0]}_"
                        f"{np.datetime_as_string(ts, 's').replace('T','_').replace(':','').replace('-','')}.flac"
                    )

                    sf.write(os.path.join(self.output_dir, out_name), audio, sr)

                except Exception as e:
                    print("NOT SELECTED:", selected_file, ts, e)

            os.remove(local_path)



