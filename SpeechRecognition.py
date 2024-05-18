import IPython
import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset



#================================================
print(torch.__version__)
print(torchaudio.__version__)

# torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)
# ==========================
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

# print("Sample Rate:", bundle.sample_rate)
#
# print("Labels:", bundle.get_labels())

# ==========================
# SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SPEECH_FILE = "slice-audios/4064.0937499999995_7472.843750000001.wav"

model = bundle.get_model().to(device)

# print(model.__class__)

# IPython.display.Audio(SPEECH_FILE)

waveform, sample_rate = torchaudio.load('slice-audios/4064.0937499999995_7472.843750000001.wav')
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

# with torch.inference_mode():
#     features, _ = model.extract_features(waveform)
#
# fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
# for i, feats in enumerate(features):
#     ax[i].imshow(feats[0].cpu(), interpolation="nearest")
#     ax[i].set_title(f"Feature from transformer layer {i+1}")
#     ax[i].set_xlabel("Feature dimension")
#     ax[i].set_ylabel("Frame (time-axis)")
# fig.tight_layout()

with torch.inference_mode():
    emission, _ = model(waveform)

# plt.imshow(emission[0].cpu().T, interpolation="nearest")
# plt.title("Classification result")
# plt.xlabel("Frame (time-axis)")
# plt.ylabel("Class")
# plt.tight_layout()
# print("Class labels:", bundle.get_labels())

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

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
