import torch
import torchaudio
from pydub import AudioSegment
import threading
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import os
import shutil
from moviepy.editor import VideoFileClip
import cloudinary
cloudinary.config(
    cloud_name = "dm7tnmhj4",
    api_key = "715756646867576",
    api_secret = "TjUu7LzW21cddLWQ45FHxm3eH3k",
    secure = True
)
import cloudinary.api
import cloudinary.uploader

class Transcription:
    def __init__(self, start_time, end_time, transcription):
        self.startTime = start_time
        self.endTime = end_time
        self.transcription = transcription

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
OUTPUT_AUDIO = "output_audio.wav"  # Path to output audio file
# SpeechRecognition set up
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

# VoiceActivityDetection set up
# login("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", add_to_git_credential=True)
sliced_audios = list()
RESULT = ""

def slice_wav():
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="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")
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(OUTPUT_AUDIO)
    for speech_turn, track, speaker in vad.itertracks(yield_label=True):
        # Slice audio
        audio = AudioSegment.from_mp3(OUTPUT_AUDIO)
        start_time = speech_turn.start * 1000
        end_time = speech_turn.end * 1000

        sliced_audio = audio[start_time:end_time]
        sliced_OUTPUT_AUDIOname = FOLDER_NAME + "/" + str(start_time) + "_" + str(end_time) + ".wav"
        sliced_audio.export(sliced_OUTPUT_AUDIOname, format="wav")

        sliced_audios.append(sliced_OUTPUT_AUDIOname)


def asr():
    open('myfile.vtt', 'w+')
    for audioSrc in sliced_audios:
        waveform, sample_rate = torchaudio.load(audioSrc)
        waveform = waveform.to(device)
    # files = os.listdir("./slice-audios")
    # for file in files:
    #     waveform, sample_rate = torchaudio.load("slice-audios/" + file)
    #     waveform = waveform.to(device)

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
        print(line_info)

        with open("myfile.vtt", "a", encoding="utf-8") as f:
            f.writelines(line_info)

def asrWithText(result):
    open('myfile.vtt', 'w+')
    for audioSrc in sliced_audios:
        waveform, sample_rate = torchaudio.load(audioSrc)
        waveform = waveform.to(device)
        # files = os.listdir("./slice-audios")
        # for file in files:
        #     waveform, sample_rate = torchaudio.load("slice-audios/" + file)
        #     waveform = waveform.to(device)

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
        print(line_info)

        with open("myfile.vtt", "a", encoding="utf-8") as f:
            f.writelines(line_info)


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

# if __name__ == "__main__":
#     video_to_audio(INPUT_VIDEO, OUTPUT_AUDIO)
#     createAudiosDir()
#     t1 = threading.Thread(target=slice_wav)
#     t2 = threading.Thread(target=asr)
#
#     t1.start()
#     t1.join()  # Wait for t1 to complete before starting t2
#     t2.start()
#
#     t2.join()
#
#     # removeAudiosDir()
#     print("Done!")
