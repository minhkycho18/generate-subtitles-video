from pydub import AudioSegment

# Load the WAV file
audio = AudioSegment.from_mp3("Y2meta.app - Time is Free but it's Priceless - Jay Shetty (128 kbps).mp3")

# Define start and end time for the slice in milliseconds
start_time = 4100
end_time = 7500

# Slice the audio
sliced_audio = audio[start_time:end_time]

# Export the sliced audio to a new WAV file
sliced_audio.export("sliced_audio.wav", format="wav")