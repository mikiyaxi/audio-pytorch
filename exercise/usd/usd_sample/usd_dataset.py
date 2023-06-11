from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import torch


class UrbanSoundDataset(Dataset):

    """few require methods are needed"""

    # constructor, takes in audio path, meta csv, and mel-spectrogram
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        target_sample_rate,
        num_samples,
        device
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir  # audio path string
        self.device = device
        self.transformation = transformation.to(self.device)  # mel, transform from waveform
        self.target_sample_rate = target_sample_rate  #
        self.num_samples = num_samples


    def __len__(self):  # take the whole length of the dataset
        return len(self.annotations)


    def __getitem__(self, index):
        """
        1) geting and loading the waveform of audio
        2) take dataset sample with index: list[1] -> list.__getitem__(1)
        """
        audio_sample_path = self._get_audio_sample_path(index)

        label = self._get_audio_sample_label(index)
        # print(label)
        # load functionality(sox or soundfile, etc) for loading torch audio
        # singal as waveform, sr stands for sample rate
        signal, sr = torchaudio.load(audio_sample_path) 
        # pass the signal tensor to GPUs for computation like resampling, or padding, etc 
        signal = signal.to(self.device)
        # -------- signal is waveform now, turn it into mel ---------
        # signal is a PyTorch Tensor -> (num_channels, samples_rate) -> (2, 16000) -> (1, 16000) channel down
        # sample_rate = 16000, means 16000 numbers form an audio file, we can also say it's 16000 samples, second name
        # before transformation make sure sample rate to stand the same

        # print("sample rate:",sr)
        signal = self._resample_if_necessary(signal, sr)
        # before transformation make sure sample are all in one channel
        signal = self._mix_down_if_necessary(signal)
        # --------- padding section ---------
        # if the input audio sample are longer than what we expected
        # we cut them

        # print("original:", signal.shape)
        signal = self._cut_if_necessary(signal)

        # print("check first time:", signal.shape)
        # if the input audio sample has less length than what we expected
        # we right padding them, [x, x, ..., x, y, y, y, ..., y], y is what we added
        signal = self._right_pad_if_necessary(signal)

        # print("check secound time:", signal.shape)

        # transformation is a function defined below
        signal = self.transformation(signal)
        # print("audio path:", audio_sample_path)
        return signal, label
 

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples): get the value of num_samples 
        if signal.shape[1] > self.num_samples:
            # for a array ":" leave the slices as it is 
            # :self.num_samples means from first index up to that 
            signal = signal[:, :self.num_samples] # simply slice the index 
        return signal 


    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # if less expcted num_samples 
            # [1, 1, 1] -> right pad -> [1, 1, 1, 0, 0, 0]
            num_missing_samples = self.num_samples - length_signal 
            # torch built-in function for padding 
            last_dim_padding = (0, num_missing_samples) # if this looks like this (1, 2)
            # [1, 1, 1] -> [0, 1, 1, 1, 0, 0]: value for padding = 0. In (1, 2) 
            # left value=1, pad one 0 on the left, similarly favor for 2 
            # in pytorch built-in method, we will pad things at the last dimension 
            # if we have last_dim_padding = (1, 1, 2, 2) like this 
            # values at index 0 and 1 are for last dimension padding, while index 2 and 3 are second to the last dimension
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        """sr is the original audio sample rate, self.target_sample_rate is what we want to normalize"""
        if sr != self.target_sample_rate:
            # define resampler
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            # getting the signal, apply resampler to the signal
            signal = resampler(signal)
        return signal


    def _mix_down_if_necessary(self, signal):
        """
        aggregarate the signal with multiple channel into one channel (mean operation)
        dim = 0 is because we have the audio prepared as this (1, 16000), one channel
        """
        if signal.shape[0] > 1:  # greater than one, not a mono channel
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal



    def _get_audio_sample_path(self, index):
        """
        1) folders of the audio file
        """
        fold = f"fold{self.annotations.iloc[index, 5]}"  # folder is at the index = 5
        path = os.path.join(
            self.audio_dir, fold, self.annotations.iloc[index, 0]
        )  # path col idx = 1
        return path


    # def _get_audio_sample_path(self, index):
    #     """
    #     1) folders of the audio file
    #     """
    #     # fold = f"fold{self.annotations.iloc[index, 5]}"  # folder is at the index = 5
    #     path = os.path.join(
    #         self.audio_dir, self.annotations.iloc[index, 0]
    #     )  # path col idx = 1
    #     return path



    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]  # label id at index = 6 for UrbanSoundDataset, 3 for esc-50


if __name__ == "__main__":
    ANNOTATIONS_FILE = "../dataset/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "../dataset/UrbanSound8K/audio/"
    # sample rate typical length for audio processing
    SAMPLE_RATE = 22050
    # an audio tensor will have an array of number/float, all those consist a audio
    # one_audio_sample = [0.001, 0.024, 0.09, ..., 0.708]
    # this is the typical way of analyze the audio, split them into smaller part
    # so basically, sample_rate & num_samples refer to the same concept
    # but they are used for different purpose, num_samples is to make the duration the same, padding audio tensor
    # because .wav file could have different length, 2s or 10s. Need to keep them as the same for matrix operation
    NUM_SAMPLES = 22050

    # send the computation to GPUs 
    if torch.cuda.is_available():
        device = "cuda"
    # elif getattr(torch, "has_mps", False):        # won't be able to do that, mps have different tensor type
    #     device = 'mps'                            # and can only run on cuda
    else:
        device = 'cpu'
    print("Using device:", device)

    # mel spectrogram transformation from .wav file
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # frame size
        hop_length=512,
        n_mels=64,  # numbers of mel
    )

    # load into the class
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE, 
                            NUM_SAMPLES,
                            device)     # which device to use, to make it working on GPU, 
                                        # you are modifying the code on self.transformation: .t0(device)
    # for checking
    # print(f"There are {len(usd)} samples in the dataset.")

    # signal, label = usd[1]  # this method is already defined, so use idx=0 slice directly

    # print("signal sample (from waveform to mel_spectrogram):", signal)
    # print("shape:", signal.shape)  # waveform: Tensor(2, 14004):
    # 2: as channels(stereo audio file)
    # 14004: number of samples in this audio file
    # right now signal has become a Tensor [1, 64, 10]
    # 1: channel
    # 64: number of mel_spectrogram
    # 10: number of frame
    # print("label:", label)
