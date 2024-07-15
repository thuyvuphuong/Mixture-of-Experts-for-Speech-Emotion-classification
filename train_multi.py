#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model, losses, optimizers
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modules import STMoE

#%%

class_mapping = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

root_path = '../CREMA-D/AudioWAV'
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
x_train, x_test, y_train, y_test = train_test_split(voices_mel_expanded, labels, test_size=0.1, random_state=42, shuffle=True)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
    
# %%
num_spatio_experts=20
num_temp_experts=20
top_k=10
inputs = tf.keras.Input(shape=x_train[0].shape)
moes_layer = STMoE(input_shape= inputs.shape, num_spatio_experts=num_spatio_experts,
                   num_temp_experts=num_temp_experts, top_k=top_k, gate_units=64)

# Apply the MoEs layer
outputs = moes_layer(inputs)
moes_model = tf.keras.Model(inputs=inputs, outputs=outputs)
moes_model.summary()

# %%
checkpoint_filepath = f'trained_models/modelspatemp_sp{num_spatio_experts}_tp{num_temp_experts}_top{top_k}.keras'
acc_save_path = f'../trained_output/acc_and_lossspatemp_sp{num_spatio_experts}_tp{num_temp_experts}_top{top_k}.jpg'
cfs_matrix_save_path = f'../trained_output/confusion_matrixspatemp_sp{num_spatio_experts}_tp{num_temp_experts}_top{top_k}.jpg'

def get_callbacks(patience = 5):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1,
                                                     patience=patience//2,
                                                     min_lr=1e-12,
                                                     mode='min',
                                                     verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    save_weights_only=False,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True,
                                                    verbose=1)
    return [reduce_lr, checkpoint]

# %%
patience = 10
epochs = 25
lr = 1e-2
batch_size = 16

callbacks = get_callbacks(patience = patience)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = ['accuracy']
moes_model.compile(optimizer, loss=loss, metrics=metrics)

# %%
history = moes_model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks = callbacks,
                    validation_split=0.1)

# %%
moes_model.load_weights(checkpoint_filepath)
y_prob = moes_model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)
if len(y_test.shape) == 1:
    y_test_onehot = tf.keras.utils.to_categorical(y_test)
    
loss, acc = moes_model.evaluate(x_test, y_test, verbose=2)

print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score: {:5.2f}".format(f1))
auc = roc_auc_score(y_test_onehot, y_prob, multi_class='ovr')
print("AUC: {:5.2f}".format(auc))
title_metrics = f"ACC: {acc:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}"

#%%
from visualize_funcs import plot_accuracy_and_loss, plot_confusion_matrix

classes_name = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

plot_accuracy_and_loss(history, acc_save_path)
plot_confusion_matrix(y_test, y_pred, title_metrics, classes_name, cfs_matrix_save_path)


# %%
