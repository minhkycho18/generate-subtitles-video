import os
import shutil

import torch
import torchaudio
from moviepy.editor import VideoFileClip
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


FOLDER_NAME = "slice-audios"
INPUT_VIDEO = "https://res.cloudinary.com/dm7tnmhj4/video/upload/v1716024510/my_folder/vphm3dbzxc9relgr6jmy.mp4"  # Path to input video file
OUTPUT_AUDIO_PATH = FOLDER_NAME + "/output_audio.wav"  # Path to output audio file
SUBTITLE_FILE_PATH = FOLDER_NAME + "/myfile.txt"

# SpeechRecognition set up
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

# VoiceActivityDetection set up
# login("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", add_to_git_credential=True)
sliced_audios = list()


def slice_wav():
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="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")
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(OUTPUT_AUDIO_PATH)
    for speech_turn, track, speaker in vad.itertracks(yield_label=True):
        # Slice audio
        audio = AudioSegment.from_mp3(OUTPUT_AUDIO_PATH)
        start_time = speech_turn.start * 1000
        end_time = speech_turn.end * 1000

        sliced_audio = audio[start_time:end_time]
        sliced_audio_name = FOLDER_NAME + "/" + str(start_time) + "_" + str(end_time) + ".wav"
        sliced_audio.export(sliced_audio_name, format="wav")

        sliced_audios.append(sliced_audio_name)


def asr(result):
    for audioSrc in sliced_audios:
        waveform, sample_rate = torchaudio.load(audioSrc)
        waveform = waveform.to(device)

        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        with torch.inference_mode():
            emission, _ = model(waveform)
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        transcript = str(decoder(emission[0]))

        transcript = transcript.lower().capitalize().replace("|", " ")
        start_time, end_time = map(float, audioSrc.replace(FOLDER_NAME + "/", "").replace(".wav", "").split('_')[:2])

        # Write file
        time_info = f"{sec2time(start_time / 1000)} --> {sec2time(end_time / 1000)}"  # subrip subtitle file time format
        text_info = transcript
        line_info = "\n" + time_info + "\n" + text_info + "\n\n"
        # print(line_info)
        result += line_info

    return result

def sec2time(sec, n_msec=3):
    if hasattr(sec, '__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        text = pattern % (h, m, s)
    else:
        text = ('%d days, ' + pattern) % (d, h, m, s)
    return text


def createAudiosDir():
    folder_path = os.getcwd() + "/" + FOLDER_NAME
    try:
        os.makedirs(folder_path)
        print("Folder created successfully.")
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")


def removeAudiosDir():
    folder_path = os.getcwd() + "/" + FOLDER_NAME

    try:
        shutil.rmtree(folder_path)
        print("Folder deleted successfully.")
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")


def video_to_audio(input_video, output_audio):
    # Load the video clip
    video_clip = VideoFileClip(input_video)

    # Extract audio
    audio_clip = video_clip.audio

    # Write audio to file
    audio_clip.write_audiofile(output_audio)