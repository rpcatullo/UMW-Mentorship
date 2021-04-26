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
      if export_filename not in os.listdir('/content/drive/MyDrive/frog_calls/Spectograms'):
        x, sr = librosa.load(os.path.join(subdirectory, filename), sr=44100)
        if len(x) < 441000:
          x = np.concatenate((x, np.zeros(shape=(441000 - len(x),))))
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(10, 10))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.savefig('/content/drive/MyDrive/frog_calls/Spectograms/' + export_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("Exporting " + export_filename + " to spectogram")
