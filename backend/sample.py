import librosa
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score
import pickle
import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv1D, Activation, Dropout, Dense, Flatten, MaxPooling1D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn.metrics as metrics
import logging
import tensorflow as tf

# Configure TensorFlow to use GPU on Mac
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info("GPU is available and configured for TensorFlow.")
    except RuntimeError as e:
        logging.error(f"Error configuring GPU: {e}")
else:
    logging.warning("GPU not found. Training will proceed on CPU.")

# Define the data directory
data_directory = '/Users/jagjeevankashid/Developer/Python/fai_project/audio_dataset'

# Ensure the dataset is in the specified directory
if not os.path.exists(data_directory):
    print(f"Data directory {data_directory} not found.")
    # Optionally, provide instructions or code to download the dataset
else:
    print(f"Data directory {data_directory} found.") 
# Emotions in the RAVDESS dataset, different numbers represent different emotion
emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def extract_feature(data, sr, mfcc, chroma, mel):
    
    """
    extract features from audio files into numpy array
    
    Parameters
    ----------
    data : np.ndarray, audio time series
    sr : number > 0, sampling rate
    mfcc : boolean, Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
    chroma : boolean, pertains to the 12 different pitch classes
    mel : boolean, Mel Spectrogram Frequency
    
    """
    
    if chroma:                          
        stft = np.abs(librosa.stft(data))  
    result = np.array([])
    if mfcc:                          
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:                          
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:                             
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T,axis=0)
        result = np.hstack((result, mel))
        
    return result 

def noise(data, noise_factor):
    
    """
    add random white noises to the audio

    Parameters
    ----------
    data : np.ndarray, audio time series
    noise_factor : float, the measure of noise to be added 

    """
    noise = np.random.randn(len(data)) 
    augmented_data = data + noise_factor * noise
    
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift(data, sampling_rate, shift_max, shift_direction):
    
    """
    shift the spectogram in a direction
    
    Parameters
    ----------
    data : np.ndarray, audio time series
    sampling_rate : number > 0, sampling rate
    shift_max : float, maximum shift rate
    shift_direction : string, right/both
    
    """
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
        
    return augmented_data

def load_data(save=False):
    
    """
    loading dataset

    Parameters
    ----------
    save : boolean, save the data to disk as .npy

    """
    x, y = [], []
    # Match both lowercase and uppercase .wav files in the correct subdirectories
    wav_files = glob.glob(os.path.join(data_directory, "Audio_Song_Actors_*", "Actor_*", "*.wav"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Speech_Actors_*", "Actor_*", "*.wav"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Song_Actors_*", "Actor_*", "*.WAV"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Speech_Actors_*", "Actor_*", "*.WAV"))
    
    # Add support for additional audio formats
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Song_Actors_*", "Actor_*", "*.mp3"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Speech_Actors_*", "Actor_*", "*.mp3"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Song_Actors_*", "Actor_*", "*.flac"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Speech_Actors_*", "Actor_*", "*.flac"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Song_Actors_*", "Actor_*", "*.ogg"))
    wav_files += glob.glob(os.path.join(data_directory, "Audio_Speech_Actors_*", "Actor_*", "*.ogg"))
    
    logging.debug(f"Number of audio files found: {len(wav_files)}")
    for file in wav_files:
        logging.debug(f"Processing file: {file}")
        try:
            data, sr = librosa.load(file)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        x.append(feature)

        file_name = os.path.basename(file)
        
        # get emotion label from the file name
        emotion = emotions.get(file_name.split("-")[2], "unknown")
        if emotion == "unknown":
            logging.warning(f"Unknown emotion label for file: {file_name}")
            continue
        y.append(emotion)

        # add noise to the data
        n_data = noise(data, 0.001)
        n_feature = extract_feature(n_data, sr, mfcc=True, chroma=True, mel=True)
        x.append(n_feature)
        y.append(emotion)

        # shift the data
        s_data = shift(data, sr, 0.25, 'right')
        s_feature = extract_feature(s_data, sr, mfcc=True, chroma=True, mel=True)
        x.append(s_feature)
        y.append(emotion)
    
    logging.info(f"Total samples collected: {len(x)}")

    if save == True:
        np.save('X', np.array(x))
        np.save('y', y)
        
    return np.array(x), y

def main():
    # Check if pre-extracted features exist
    if os.path.exists('X.npy') and os.path.exists('y.npy'):
        logging.info("Loading pre-extracted features from disk.")
        X = np.load('X.npy')
        y = np.load('y.npy')
    else:
        X, y = load_data(save=True)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    
    labelencoder = LabelEncoder()
    labelencoder.fit(y_train)
    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    logging.info(le_name_mapping)
    
    y_train = labelencoder.transform(y_train)
    y_test = labelencoder.transform(y_test)
    
    # Get the number of features extracted
    logging.info(f'Features extracted: {x_train.shape[1]}')
    
    model = Sequential()
    model.add(Conv1D(256, 5,padding='same', input_shape=(180,1))) # 1st layer
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) # 2nd layer
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) # 3rd layer
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) # 4th layer
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=8,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)
                    )
    ) # 7th layer
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(decay=1e-6)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    XProccessed = np.expand_dims(x_train, axis=2)
    XTestProcessed = np.expand_dims(x_test, axis=2)
    history = model.fit(XProccessed, y_train, epochs=100, validation_data=(XTestProcessed, y_test), batch_size=64)
    model.save('trained_model.h5')
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate the model
    y_pred = model.predict(XTestProcessed)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate classification report
    class_report = metrics.classification_report(y_test, y_pred_classes, target_names=emotions.values())
    logging.info("Classification Report:\n" + class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=emotions.values(),
               yticklabels=emotions.values())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    main()