import IPython
import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

import os


torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
SPEECH_FILE = "slice-audios/4064.0937499999995_7472.843750000001.wav"

model = bundle.get_model().to(device)



class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


# files = os.listdir("./slice-audios")
# for file in files:
#     waveform, sample_rate = torchaudio.load("slice-audios/" + file)
#     waveform = waveform.to(device)
#
#     if sample_rate != bundle.sample_rate:
#         waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
#
#     with torch.inference_mode():
#         emission, _ = model(waveform)
#     decoder = GreedyCTCDecoder(labels=bundle.get_labels())
#     transcript = decoder(emission[0])
#
#     print(transcript)

import os

# Get the current working directory
current_directory = os.getcwd()

print("Current directory:", current_directory)