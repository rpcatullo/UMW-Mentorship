import os
import matplotlib.pyplot as plt
import numpy as np

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd

directory = '/content/drive/MyDrive/frog_calls'
for foldername in os.listdir(directory):
  subdirectory = '/content/drive/MyDrive/frog_calls/' + str(foldername) 
  for filename in os.listdir(subdirectory):
    if "Unidentified" not in subdirectory and ".ipynb_checkpoints" not in filename:
      export_filename = filename[:len(filename) - 4] + '.png'
      if export_filename not in os.listdir('/content/drive/MyDrive/frog_calls/Spectrograms'):
        x, sr = librosa.load(os.path.join(subdirectory, filename), sr=44100)
        if len(x) == 176400:
          sgram = librosa.stft(x)
          sgram_mag, _ = librosa.magphase(sgram)
          mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
          mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
          plt.figure(figsize=(10, 10))
          librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel')
          plt.axis('off')
          plt.savefig('/content/drive/MyDrive/frog_calls/Spectrograms/' + export_filename, bbox_inches='tight', pad_inches=0)
          plt.close()
          print("Exporting " + export_filename + " to mel scale spectrogram")
