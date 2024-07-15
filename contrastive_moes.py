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
from tensorflow.keras import backend as K
from keras import ops
from modules import MoEMLP

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
def euclidean_distance(vects):
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, tf.keras.backend.epsilon()))

# %%
num_experts = 10
top_k = 5
filters = 128
inputs = tf.keras.Input(shape=x_train[0].shape)


x = layers.BatchNormalization()(inputs)
x = layers.Conv2D(8, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="tanh")(x)
outputs = layers.Dense(1024, activation="tanh")(x)
embedding_network = tf.keras.Model(inputs=inputs, outputs=outputs)
embedding_network.summary()

# %%
input_1 = tf.keras.layers.Input((128, 118, 1))
input_2 = tf.keras.layers.Input((128, 118, 1))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = tf.keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
    [tower_1, tower_2]
)
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)

# %%
def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

# %%

siamese.summary()

checkpoint_filepath = f'trained_models/contrastive_embedding.keras'

# %%
epochs = 30
patience = 10
lr = 1e-2
batch_size = 32
siamese.compile(loss=loss(margin=1), optimizer="Adam", metrics=["accuracy"])

# %%
siamese.load_weights(checkpoint_filepath)

# %%
siamese.trainable = False
layers = siamese.layers

# Extract the first three layers
input_layer = layers[0]
first_layer = layers[1]
second_layer = layers[2]


# %%
num_experts = 50
top_k = 30
inputs = tf.keras.Input(shape=x_train[0].shape)
embedding_features = second_layer(inputs)
moes_layer = MoEMLP(num_experts=num_experts, 
                    dense_units=512,
                    top_k = top_k, dropout_rate=0)

outputs = moes_layer(embedding_features)
moes_model = Model(inputs=inputs, outputs=outputs)

# Summary of the embedding model
moes_model.summary()
# %%
checkpoint_filepath = f'trained_models/contrastive_moes{num_experts}_top{top_k}.keras'
acc_save_path = f'../trained_output/contrastive_moes{num_experts}_top{top_k}.jpg'
cfs_matrix_save_path = f'../trained_output/contrastive_moes{num_experts}_top{top_k}.jpg'

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
epochs = 100
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
