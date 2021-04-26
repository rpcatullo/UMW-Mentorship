from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
import wavio
import os
import contextlib
import wave
import librosa

# Chunk audio files longer than 10 seconds into separate files

# Iterate through all folders in the frog calls dataset
directory = '/content/drive/MyDrive/frog_calls'
for foldername in os.listdir(directory):
  subdirectory = '/content/drive/MyDrive/frog_calls/' + str(foldername) 
  for filename in os.listdir(subdirectory):
    # Ignore unidentified long recordings, as they don't have labels
    if "Unidentified" not in subdirectory and ".ipynb_checkpoints" not in filename:
      print(os.path.join(subdirectory, filename))
      with contextlib.closing(wave.open(os.path.join(subdirectory, filename),'r')) as f:
        # Calculate the duration of the audio file
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        # Chunk the audio into 10 second chunks
        if duration > 10:
          myaudio = AudioSegment.from_file(os.path.join(subdirectory, filename), "wav") 
          chunk_length_ms = 10000 # pydub calculates in millisec
          chunks = make_chunks(myaudio, chunk_length_ms) # Make chunks of 10 seconds
          for i, chunk in enumerate(chunks):
            # Export the 10 second chunks as .wav files
            chunk_name = filename[:len(filename)-4] + ".{0}.wav".format(i)
            print("exporting" + " " + chunk_name)
            chunk.export(subdirectory + "/" + chunk_name, format="wav")
          # Remove the original file
          os.remove(subdirectory + "/" + filename)
