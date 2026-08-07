import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import librosa
import torch


def load_conformer_model():
    from Encoder import Encoder
    model_path = '../../models/Conformer_Embedding.ckpt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = Encoder(Baseline=False, num_classes=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class Conformer():
    def __init__(self, loaded_model, audio_path, samplerate, window_size):
        self.audio_path = audio_path
        self.sample_rate = samplerate
        self.window_size = window_size
        self.model = loaded_model

    def run_Conformer_model(self, sample):
        print('CONFORMER')
        return self.model(torch.from_numpy(np.expand_dims(sample, axis=1)))

    def _load_audio(self, audio_file):
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        return audio

    def _window_audio(self, audio):
        residue = len(audio) % (self.window_size * self.sample_rate)
        if residue != 0:
            audio = audio[:-residue].copy()
        samples = np.split(audio, len(audio) / (self.window_size * self.sample_rate))
        return samples

    def __call__(self):
        for file in os.listdir(self.audio_path):
            if file.endswith('.flac') or file.endswith('.wav'):
                audio = self._load_audio(os.path.join(self.audio_path, file))
                try:
                    windowed_audio = self._window_audio(audio)
                except Exception as e:
                    print('Conformer exception: ', str(e))
                    continue
                if np.array(windowed_audio).shape[0] > 100:
                    sub_arrays = np.array_split(np.array(windowed_audio), int(len(windowed_audio)/100))
                    for array in sub_arrays:
                        _, sub_embedding = self.run_Conformer_model(array)
                        try:
                            embeddings = torch.cat([embeddings, sub_embedding], 0)
                        except:
                            embeddings = sub_embedding
                else:
                    _, embeddings = self.run_Conformer_model(np.array(windowed_audio))
                try:
                    windowed_embeddings = torch.concat([windowed_embeddings, embeddings], 0)
                except:
                    windowed_embeddings = embeddings
        return windowed_embeddings

