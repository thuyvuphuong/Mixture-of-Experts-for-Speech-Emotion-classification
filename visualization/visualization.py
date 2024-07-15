#%%
import matplotlib.pyplot as plt
import os
import librosa
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import pandas as pd

#%%
class_mapping = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

root_path = 'CREMA-D/AudioWAV'
audio_files = os.listdir(root_path)
audio_files[:10]

def get_label(filename):
    label = filename.split('_')
    return label[2]

def process_file(audio_file, target_length=60000):
    label = get_label(audio_file)
    _, audio = wavfile.read(os.path.join(root_path, audio_file))
    audio = np.array(audio, dtype='float32')
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]
    S = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return label, S_dB

#%%

# Initialize lists to store classes and MFCCs
voices_mel = []
voices_raw = []
labels = []

for i in tqdm(range(len(audio_files))):
    class_name, mel = process_file(audio_files[i])
    labels.append(class_mapping.get(class_name))
    voices_mel.append(mel)

voices_mel = np.array(voices_mel)
labels = np.array(labels)

print(voices_mel.shape)
print(labels.shape)

voices_mel_expanded = np.expand_dims(voices_mel, axis=-1)
voices_raw_expanded = np.expand_dims(voices_raw, axis=-1)
x_train, x_test, y_train, y_test = train_test_split(voices_mel_expanded, labels, test_size=0.2, random_state=42, shuffle=True)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(voices_mel, labels, class_mapping):
    plt.figure(figsize=(14, 7))

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    for i, class_index in enumerate(unique_classes):
        class_samples = voices_mel[labels == class_index]
        sample_index = np.random.randint(0, len(class_samples))
        sample = class_samples[sample_index]

        # Adjust class index if it starts from 0
        class_key = class_index if class_index in class_mapping.values() else class_index + 1

        plt.subplot(2, 3, i + 1)
        plt.imshow(sample, cmap='jet', origin='lower')
        # plt.title('Class: {}'.format(class_mapping[class_key]))
        plt.colorbar()

    plt.tight_layout()
    plt.show()

# Assuming 'voices_mel' and 'labels' are already defined
visualize_samples(voices_mel, labels, class_mapping)


# %%
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Function to read and plot audio file
def plot_audio(file_path, subplot_position, title):
    sample_rate, data = wavfile.read(file_path)
    plt.subplot(6, 1, subplot_position)
    plt.plot(data)
    plt.title(title, fontsize=13)
    plt.grid()

# Set up the figure
plt.figure(figsize=(12, 14))

# Plot each audio file
plot_audio('CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav', 1, 'Ang')
plot_audio('CREMA-D/AudioWAV/1001_DFA_DIS_XX.wav', 2, 'Dis')
plot_audio('CREMA-D/AudioWAV/1001_DFA_FEA_XX.wav', 3, 'Fea')
plot_audio('CREMA-D/AudioWAV/1001_DFA_HAP_XX.wav', 4, 'Hap')
plot_audio('CREMA-D/AudioWAV/1001_DFA_NEU_XX.wav', 5, 'Neu')
plot_audio('CREMA-D/AudioWAV/1001_DFA_SAD_XX.wav', 6, 'Sad')

# Display the plots
plt.tight_layout()
plt.show()

# %%
from sklearn.decomposition import PCA
voices_mel_flattened = voices_mel.reshape(len(voices_mel), -1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(voices_mel_flattened, labels, test_size=0.2, random_state=42, shuffle=True)

# Apply PCA to the entire dataset
pca = PCA(n_components=2)
voices_pca = pca.fit_transform(voices_mel_flattened)

# Split PCA-transformed data into training and testing sets
x_train_pca = voices_pca[:len(x_train)]
x_test_pca = voices_pca[len(x_train):]

# Visualize PCA features for both training and testing data in separate subplots
plt.figure(figsize=(10, 4))

# Subplot for training data
plt.subplot(1, 2, 1)
for label in np.unique(y_train):
    plt.scatter(x_train_pca[y_train == label, 0], x_train_pca[y_train == label, 1], 
                label=list(class_mapping.keys())[list(class_mapping.values()).index(label)], alpha=0.5)
plt.title('PCA of Training Data Mel Spectrograms', fontsize=13)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()

# Subplot for testing data
plt.subplot(1, 2, 2)
for label in np.unique(y_test):
    plt.scatter(x_test_pca[y_test == label, 0], x_test_pca[y_test == label, 1], 
                label=list(class_mapping.keys())[list(class_mapping.values()).index(label)], alpha=0.5)
plt.title('PCA of Testing Data Mel Spectrograms', fontsize=13)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# %%
