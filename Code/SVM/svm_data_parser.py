import os
import numpy as np
import pandas as pd
import librosa as lb
from mutagen.wave import WAVE
from tqdm import tqdm


class SVMData:
    def __init__(self, patient_data_path, audio_data_path, *args, **kwargs):
        self.patient_data_path = patient_data_path
        self.audio_data_path = audio_data_path

        whitelist = ['patient_data', 'audio_data', 'task_mapping']
        for key, val in kwargs.items():
            if key in whitelist:
                setattr(self, key, val)

    def set_task_mapping(self, task_mapping):
        self.task_mapping = task_mapping

    def parse_patients(self, include_control=True):
        patients = pd.read_excel(self.patient_data_path)
        depression_only = patients.loc[
            (patients['thought.disorder.symptoms'] == 0.) &
            (patients['depression.symptoms'] != 0.)]
        assert depression_only.loc[depression_only['thought.disorder.symptoms'] != 0.].empty

        if include_control:
            control_group = patients.loc[
                (patients['depression.symptoms'] == 0.) &
                (patients['thought.disorder.symptoms'] == 0.)]
            self.patient_data = pd.concat([depression_only, control_group])
        else:
            self.patient_data = depression_only

        self.patient_data.drop(['education.level', 'diagnosis', 'thought.disorder.symptoms', 'group'], axis=1, inplace=True)
        self.patient_data.sex.replace(['female', 'male'], [0, 1], inplace=True)
        self.patient_data.age.fillna(self.patient_data.age.mean(), inplace=True)
        self.patient_data.age = self.patient_data.age.astype('int64')

        return self.patient_data

    def parse_audio(self, save=False, save_path='./svm_df.pkl'):
        self.audio_data = self.patient_data.copy()
        self.audio_data['audio'] = self.audio_data.apply(self._get_patient_audio, axis=1)
        # exclude patients with no recordings
        self.audio_data = self.audio_data[self.audio_data.audio.apply(len) == 3]
        for domain in self.task_mapping:
            self.audio_data[f'audio.{domain}'] = self.audio_data.apply(self._get_domain_audio, axis=1, domain=domain)

        narr_coeffs = self._process_recordings(self.audio_data['audio.narrative']).add_prefix('narr_')
        story_coeffs = self._process_recordings(self.audio_data['audio.story']).add_prefix('story_')
        inst_coeffs = self._process_recordings(self.audio_data['audio.instruction']).add_prefix('inst_')
        self.audio_data = self.audio_data.reset_index()
        self.audio_data = pd.concat([self.audio_data, narr_coeffs + 80, story_coeffs + 80, inst_coeffs + 80], axis=1)
        self.audio_data.drop(
            ['ID', 'index'] + [col for col in self.audio_data if col.startswith('audio')],
            axis=1,
            inplace=True)

        if save:
            self.audio_data.to_pickle(save_path)
        return self.audio_data

    def _get_domain_audio(self, row, domain):
        """
        Get recordings of particular patient
        """
        files = []
        for topic in self.task_mapping[domain]:
            for file_name in row.audio:
                if file_name.find(topic) != -1:
                    files.append(file_name)

        assert len(files) < 2
        return files[0] if len(files) else None

    def _get_patient_audio(self, row):
        """
        Find names of patient's recordings
        """
        key = row.ID
        audio_files = []
        for filename in os.listdir(self.audio_data_path):
            if filename.find(key) != -1:
                audio_files.append(filename)
        return audio_files

    def _get_cutoff_duration(self, files):
        """
        Find cutoff qunatile for long recordings
        """
        lengths = []
        for _, filename in files.items():
            if filename is not None:
                audio = WAVE(os.path.join(self.audio_data_path, filename))
                lengths.append(int(audio.info.length))
        lengths = np.asarray(lengths)
        duration_limit = np.quantile(lengths, [0.9])
        return duration_limit

    def _get_spectrogram_coeffs(self, files, duration):
        """
        Dervive coefficients for recordings column
        """
        for _, filename in tqdm(files.items(), total=len(files)):
            signal, sr = lb.load(os.path.join(self.audio_data_path, filename), sr=12000)

            # this is the number of samples in a window per fft
            n_fft = int(sr * 0.3)  # 30 ms

            # The amount of samples we are shifting after each fft
            hop_length = int(n_fft / 2)

            mel_signal = lb.feature.melspectrogram(
                y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=48
            )

            spectrogram = np.abs(mel_signal)
            power_to_db = lb.power_to_db(spectrogram, ref=np.max)

            # pad with zeroes or cut
            pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack(
                (a, np.full((a.shape[0], i - a.shape[1]), -80.0)))
            power_to_db = pad2d(power_to_db, i=int(duration * sr / hop_length))

            # unroll spectrogram
            coeffs_arr = power_to_db.reshape(-1)
            yield coeffs_arr

    def _process_recordings(self, files):
        """
        Create new dataframe with coefficients for a given column
        """
        duration = self._get_cutoff_duration(files)
        coeffs_gen = self._get_spectrogram_coeffs(files, duration)
        return pd.DataFrame(coeffs_gen)



task_mapping = {
    'narrative': ['sportsman', 'adventure', 'winterday'], 
    'story': ['present', 'trip', 'party'], 
    'instruction': ['chair', 'table', 'bench']
}

svm_data = SVMData(
    '/Users/koldi/se/DepressionPrediction/Psychiatric-Disorders-Data/Notebooks/psychiatric_disorders_data.ML/PsychiatricDiscourse_participant.data.xlsx',
    '/Users/koldi/se/DepressionPrediction/Psychiatric-Disorders-Data/Notebooks/psychiatric_disorders_data.ML/wav files',
    task_mapping=task_mapping
)

svm_data.parse_patients()
svm_data.parse_audio(save=True)

