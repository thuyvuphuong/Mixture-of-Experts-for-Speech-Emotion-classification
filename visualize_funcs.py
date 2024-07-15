import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns


def plot_accuracy_and_loss(history, save_path=None):
    """
    Plots training and validation accuracy and loss from a Keras history object.
    
    Args:
    history: Keras History object.
    save_path: Path to save the plot. If None, the plot will be displayed.
    """
    val_acc = history.history['val_accuracy']
    train_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']

    epochs = range(1, len(val_acc) + 1)

    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='blue')
    plt.plot(epochs, train_acc, label='Training Accuracy', color='orange')
    plt.fill_between(epochs, val_acc, train_acc, where=[v > t for v, t in zip(val_acc, train_acc)], interpolate=True, color='blue', alpha=0.3)
    plt.fill_between(epochs, val_acc, train_acc, where=[v < t for v, t in zip(val_acc, train_acc)], interpolate=True, color='orange', alpha=0.3)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Validation', 'Training'])
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='blue')
    plt.plot(epochs, train_loss, label='Training Loss', color='orange')
    plt.fill_between(epochs, val_loss, train_loss, where=[v > t for v, t in zip(val_loss, train_loss)], interpolate=True, color='blue', alpha=0.3)
    plt.fill_between(epochs, val_loss, train_loss, where=[v < t for v, t in zip(val_loss, train_loss)], interpolate=True, color='orange', alpha=0.3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Validation', 'Training'])
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, title:str, class_names, save_path=None):
    """
    Plots the confusion matrix using true and predicted labels.
    
    Args:
    y_true: Array of true labels.
    y_pred: Array of predicted labels.
    class_names: List of class names.
    save_path: Path to save the plot. If None, the plot will be displayed.
    """
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    cm_per = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100,2)
    # cm_per=cm
    df_cm = pd.DataFrame(cm_per, index = class_names, columns = class_names)
    plt.figure(figsize = (7, 6))    
    sns.heatmap(df_cm, annot=True, cmap = "Blues", linewidths=.1 ,fmt='.2f')
    plt.title(title)
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
def visualize_data_distribution(y_train, y_test, class_mapping):
    inverse_class_mapping = {v: k for k, v in class_mapping.items()}
    
    train_labels_df = pd.DataFrame(y_train, columns=['Emotion'])
    test_labels_df = pd.DataFrame(y_test, columns=['Emotion'])

    train_labels_df['Emotion'] = train_labels_df['Emotion'].map(inverse_class_mapping)
    test_labels_df['Emotion'] = test_labels_df['Emotion'].map(inverse_class_mapping)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(y='Emotion', data=train_labels_df, order=train_labels_df['Emotion'].value_counts().index, hue='Emotion', palette='terrain', legend=False)
    plt.title('Number of Samples per Class in Training Set', fontsize=13)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    
    # Adding value counts on the bars
    # for i, v in enumerate(train_labels_df['Emotion'].value_counts()):
    #     plt.text(v + 10, i, str(v), ha='left', va='center', color='black', fontsize=12)

    plt.subplot(1, 2, 2)
    sns.countplot(y='Emotion', data=test_labels_df, order=test_labels_df['Emotion'].value_counts().index, hue='Emotion', palette='terrain', legend=True)
    plt.title('Number of Samples per Class in Testing Set', fontsize=13)
    plt.xlabel('Count')
    plt.ylabel('Emotion')
    
    # Adding value counts on the bars
    # for i, v in enumerate(test_labels_df['Emotion'].value_counts()):
    #     plt.text(v + 10, i, str(v), ha='left', va='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()
    


# Example usage
# Assuming `history` is the history object returned by the `fit` method in Keras
# plot_accuracy_and_loss(history)

# Assuming `moes_model` is your trained model and `x_test`, `y_test` are your test data
# y_prob = moes_model.predict(x_test)
# y_pred = np.argmax(y_prob, axis=1)
# classes_name = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
# plot_confusion_matrix(y_test, y_pred, classes_name)
