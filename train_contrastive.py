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
from keras import ops

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

#%%
def make_pairs(x, y):
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = np.random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = np.random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = np.random.randint(0, num_classes - 1)

        idx2 = np.random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

#%%

pairs_train, labels_train = make_pairs(x_train, y_train)
pairs_test, labels_test = make_pairs(x_test, y_test)
x_train_1 = pairs_train[:, 0] 
x_train_2 = pairs_train[:, 1]
x_test_1 = pairs_test[:, 0] 
x_test_2 = pairs_test[:, 1]

# %%
def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    num_row = to_show // num_col if to_show // num_col != 0 else 1
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()
    
# %%
visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)

# %%
def euclidean_distance(vects):
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, tf.keras.backend.epsilon()))

# %%
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
epochs = 100
patience = 10
lr = 1e-2
batch_size = 32
callbacks = get_callbacks(patience = patience)
siamese.compile(loss=loss(margin=1), optimizer="Adam", metrics=["accuracy"])

# %%
history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks
)
# %%
siamese.load_weights(checkpoint_filepath)
results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)

# %%

def plt_metric(history, metric, title, save_path, has_valid=True):
    plt.figure()
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

# Example usage:
# Plot and save the accuracy plot
plt_metric(history=history.history, metric="accuracy", title="Model accuracy", save_path="../trained_output/contrastive_model_accuracy.png")

# Plot and save the contrastive loss plot
plt_metric(history=history.history, metric="loss", title="Contrastive Loss", save_path="../trained_output/contrastive_loss.png")


# %%
