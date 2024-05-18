import torchaudio
from mpmath.identification import transforms
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from huggingface_hub import login
from torchaudio.sox_effects import apply_effects_tensor

AUDIO_FILE = "Y2meta.app - Time is Free but it's Priceless - Jay Shetty (128 kbps).mp3"
login("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", add_to_git_credential=True)

# model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="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")
# pipeline = VoiceActivityDetection(segmentation=model)
# HYPER_PARAMETERS = {
#     # remove speech regions shorter than that many seconds.
#     "min_duration_on": 0.0,
#     # fill non-speech regions shorter than that many seconds.
#     "min_duration_off": 0.0
# }
# pipeline.instantiate(HYPER_PARAMETERS)
# vad = pipeline(AUDIO_FILE)
# for speech_turn, track, speaker in vad.itertracks(yield_label=True):
#     print(f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}")
#     print(type(speech_turn.start))

waveform, sample_rate = torchaudio.load(AUDIO_FILE, normalize=True)
waveform_reversed, sample_rate = apply_effects_tensor(waveform, sample_rate, [["reverse"]])
transform = transforms.Vad(sample_rate=sample_rate, trigger_level=7.5)
waveform_reversed_front_trim = transform(waveform_reversed)
waveform_end_trim, sample_rate = apply_effects_tensor(waveform_reversed_front_trim, sample_rate, [["reverse"]])