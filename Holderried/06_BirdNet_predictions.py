# %% [markdown]
# ###Env to use: birdnet

# %%
import librosa
from skimage import feature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import os
from pathlib import Path
from sklearn.metrics import average_precision_score


# %%
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

# %% [markdown]
# Predict for positive segments first

# %%
wav_directory='../data/Holderried/selections_wavs/1/'
wav_files = os.listdir(wav_directory)

# %%
predictions = dict.fromkeys(wav_files, 0.0)

# %%
for file in wav_files:
    audio_path = Path(os.path.join(wav_directory, file))
    recording = Recording(
        analyzer,
        audio_path,
        min_conf=0.0
    )
    recording.analyze()
    for detection in recording.detections:
        if detection['scientific_name']=='Scolopax rusticola':
            predictions[file] = detection['confidence']

# %%
np.save('holderried_predictions_selections.npy', predictions)

# %%
np.mean(list(predictions.values()))

# %% [markdown]
# Predict for negative segments

# %%
wav_directory='../data/test_data_3/audio/0/'
wav_files = os.listdir(wav_directory)

# %%
predictions = dict.fromkeys(wav_files, 0.0)

# %%
for file in wav_files:
    audio_path = Path(os.path.join(wav_directory, file))
    recording = Recording(
        analyzer,
        audio_path,
        min_conf=0.0
    )
    recording.analyze()
    for detection in recording.detections:
        if detection['scientific_name']=='Scolopax rusticola':
            predictions[file] = detection['confidence']

# %%
np.save('holderried_preds_testdata3_neg.npy', predictions)

# %% [markdown]
# ##### !!! Watch out the script changed, Negatives are from different source now

# %%
sorted_keys = sorted(predictions.keys())

# Step 2: Extract scores and generate true labels
y_pred = [predictions[k] for k in sorted_keys]
y_true = [0 if len(k)>9 else 1 for k in sorted_keys]

# Step 3: Compute Average Precision
ap = average_precision_score(y_true, y_pred)
ap

# %%
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Only for binary classification (adjust for multi-class)
precision, recall, thr = precision_recall_curve(y_true, y_pred)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# %%
plt.plot(thr,precision[:-1])
plt.ylabel("Precision")
plt.xlabel("Threshold")

# %%
plt.plot(thr, recall[:-1])
plt.ylabel("Recall")
plt.xlabel("Threshold")

# %%
#preds_negative = {k: predictions[k] for k in wav_files if k in predictions}
#preds_negative = np.array(list(preds_negative.values()))

# %%

# Sample data
x = thr
y1 = precision[:-1]            # First curve
y2 = recall[:-1]      # Second curve

# Create figure and first axis
fig, ax1 = plt.subplots()

# Plot the first line
color1 = 'tab:blue'
ax1.set_xlabel('Confidence threshold')
ax1.set_ylabel('Precision', color=color1)
ax1.plot(x, y1, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True) 
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.05])

# Create second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Recall', color=color2)
ax2.plot(x, y2, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Optional: improve layout
fig.tight_layout()

plt.show()


# %%
f1 = 2*precision*recall/(precision+recall)

# %%
plt.plot(thr, f1[:-1], '-m')
plt.ylabel("f1")
plt.xlabel("Threshold")
plt.grid()

# %%
np.max(f1)

# %% [markdown]
# Map the values to uniformly spaced threshold values

# %%
from scipy.interpolate import interp1d

# %%
#thr_uniform = np.linspace(0, 1, 101)
thr_uniform = np.linspace(thr.min(), thr.max(), 100)

# %%
interpolator = interp1d(thr, f1[:-1], kind='linear')  # or 'cubic', 'quadratic', etc.
f1_score = interpolator(thr_uniform)

# %%
np.savez('holderried_f1.npz', thr=thr_uniform, f1=f1_score)

# %%
plt.plot(thr_uniform,f1_score)
plt.ylabel("f1")
plt.xlabel("Threshold")
plt.grid()

# %%



