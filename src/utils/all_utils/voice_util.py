import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Tuple
import matplotlib.pyplot as plt
import torchaudio
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
import base64
import os



class AudioProcessing:

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> None:
        """
        Converts the given audio file to WAV format with a sample rate of 16kHz and mono channel.
        :param input_path: Path to the input audio file.
        :param output_path: Path to the output WAV file.
        """
        # Load audio file
        y, sr = librosa.load(input_path, sr=None)

        # Convert to mono if necessary
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Convert to 16kHz if necessary
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # Save to WAV file
        wavfile.write(output_path, 16000, y)


    @staticmethod
    def extract_mfcc(audio_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extracts MFCC features from the given audio signal with the specified sample rate.
        :param audio_signal: The audio signal.
        :param sample_rate: The sample rate of the signal.
        :return: The MFCC features.
        """
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)

        return mfccs


    @staticmethod
    def calculate_zero_crossing_rate(audio_signal: np.ndarray) -> float:
        """
        Calculates the zero-crossing rate of the given audio signal.
        :param audio_signal: The audio signal.
        :return: The zero-crossing rate.
        """
        # Calculate zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_signal)

        return np.mean(zcr)


    @staticmethod
    def detect_voice_activity(audio_signal: np.ndarray, sample_rate: int,
                             threshold: float) -> List[Tuple[int, int]]:
        """
        Detects voice activity in the given audio signal with the specified sample rate and threshold.
        :param audio_signal: The audio signal.
        :param sample_rate: The sample rate of the signal.
        :param threshold: The threshold for voice activity detection.
        :return: A list of tuples, where each tuple contains the start and end indices of a voice activity segment.
        """
        # Compute energy
        energy = librosa.feature.rms(y=audio_signal)

        # Apply threshold
        voice_activity = np.greater(energy, threshold)

        # Find segments
        voice_activity_segments = librosa.util.frame(voice_activity, frame_length=100, hop_length=50)

        # Find segments that are above the threshold
        voice_activity_segments = [(start, end) for start, end in zip(np.where(voice_activity_segments)[0], np.where(voice_activity_segments)[0] + np.where(voice_activity_segments)[1]) if np.mean(energy[start:end]) > threshold]

        return voice_activity_segments
    



    def normalize_length(self, signal, length):
        """
        Normalizes the length of an audio signal to the desired length.

        Parameters:
            signal (numpy.ndarray): The input audio signal.
            length (int): The desired length of the audio signal.

        Returns:
            numpy.ndarray: The normalized audio signal.

        """
        if len(signal) > length:
            # Truncate the signal
            signal = signal[:length]
        elif len(signal) < length:
            # Pad the signal with zeros
            signal = np.pad(signal, (0, length - len(signal)), mode='constant')
        return signal

    # def visualize_spectrogram(self, signal, sample_rate=16000):
    #     """Visualizes the spectrogram of an audio signal."""
    #     # Compute the spectrogram
    #     D = librosa.amplitude_to_db(librosa.stft(signal), ref=np.max)

    #     # Create a figure and axis for the spectrogram
    #     plt.figure(figsize=(10, 4))
    #     plt.imshow(D, cmap='gray_r', aspect='auto')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.title('Spectrogram')
    #     plt.xlabel('Time')
    #     plt.ylabel('Frequency')
    #     plt.tight_layout()
    #     plt.show()


    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initializes the x-vector model and sets it to the specified device.
        :param model_path: Path to the x-vector model.
        :param device: The device to set the model to.
        """
        self.model = torchaudio.models.create_kaldi_xvector_model(model_path)
        self.model = self.model.to(device)

    def generate_embedding(self, audio_file_path: str) -> torch.Tensor:
        """
        Generates the x-vector embedding for the given audio file.
        :param audio_file_path: Path to the audio file.
        :return: The x-vector embedding.
        """
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # Convert to mono if necessary
        if waveform.size(-1) > 1:
            waveform = torch.mean(waveform, dim=-1)

        # Normalize audio
        waveform = waveform / torch.norm(waveform)

        # Generate embedding
        with torch.no_grad():
            embedding = self.model.embed_utterance(waveform)[0]

        return embedding
    

    @staticmethod
    def compute_cosine_similarity(file1_path, file2_path):
        """
        Computes the cosine similarity between two audio files.
        :param file1_path: Path to the first audio file.
        :param file2_path: Path to the second audio file.
        :return: The cosine similarity between the two audio files.
        """
        # Load audio files
        y1, sr1 = librosa.load(file1_path, sr=None)
        y2, sr2 = librosa.load(file2_path, sr=None)

        # Convert to mono if necessary
        if y1.ndim > 1:
            y1 = np.mean(y1, axis=1)
        if y2.ndim > 1:
            y2 = np.mean(y2, axis=1)

        # Convert to 16kHz if necessary
        sr = max(sr1, sr2)
        if sr1 != sr:
            y1 = librosa.resample(y1, sr1, sr)
        if sr2 != sr:
            y2 = librosa.resample(y2, sr2, sr)

        # Extract MFCC features
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)

        # Calculate mean vectors
        mean_mfcc1 = np.mean(mfcc1, axis=1)
        mean_mfcc2 = np.mean(mfcc2, axis=1)

        # Calculate cosine similarity
        cosine_similarity = np.dot(mean_mfcc1, mean_mfcc2) / (np.linalg.norm(mean_mfcc1) * np.linalg.norm(mean_mfcc2))

        return cosine_similarity

    def split_signal_by_timeframe(self, signal, time_frame):
        """
        Splits the signal into chunks of a specified time frame.
        :param signal: The signal to split.
        :param time_frame: The time frame of each chunk in seconds.
        :return: A list of tuples, where each tuple contains the start and end time of each chunk in seconds.
        """
        chunk_size = int(self.sample_rate * time_frame)
        chunks = [signal[i:i + chunk_size] for i in range(0, len(signal), chunk_size)]
        return [(i * self.sample_rate, (i + 1) * self.sample_rate) for i in range(len(chunks))]

    


# WavConverter:
# Input: input_path (string) - path to the input audio file, output_path (string) - path to the output WAV file.
# Output: None (it saves the converted audio file to the specified output path).
        

# MFCCExtractor:
# Input: audio_signal (numpy array) - the audio signal, sample_rate (integer) - the sample rate of the audio signal.
# Output: mfccs (numpy array) - the extracted MFCC features.


# ZeroCrossingRateCalculator:
# Input: audio_signal (numpy array) - the audio signal.
# Output: zcr (float) - the calculated zero-crossing rate.


# VoiceActivityDetector:
# Input: audio_signal (numpy array) - the audio signal, sample_rate (integer) - the sample rate of the audio signal, threshold (float) - the threshold for voice activity detection.
# Output: voice_activity_segments (list of tuples) - a list of tuples, where each tuple contains the start and end indices of a voice activity segment.


# SignalProcessor:

# normalize_length:
# Input: signal (numpy array) - the audio signal, length (integer) - the desired length of the audio signal.
# Output: normalized_signal (numpy array) - the normalized audio signal.


# visualize_spectrogram:
# Input: signal (numpy array) - the audio signal, sample_rate (integer, optional) - the sample rate of the audio signal (default: 16000 Hz).
# Output: None (it displays the spectrogram of the input audio signal).
    
# XVectorEmbeddingGenerator:
# Input: input_path (string) - path to the input audio file
# Output : PyTorch tensor with shape (1, embedding_size), where embedding_size is the size of the x-vector embedding.


# CosineSimilarity
#Input:
# file1_path: a string representing the path to the first audio file
# file2_path: a string representing the path to the second audio file
# Output:
# cosine_similarity: a float representing the cosine similarity between the two audio files. The value will be in the range of -1 to 1, 
# where 1 indicates that the two audio files are identical, 0 indicates that they are uncorrelated, and -1 indicates that they are completely 
# opposite.

def convert_binary_to_base64(binary_data):
    """
    Converts binary data to a base 64 string.

    Parameters:
    binary_data (bytes): The binary data to be converted.

    Returns:
    str: The base 64 string representation of the binary data.
    """ 
    return base64.b64encode(binary_data).decode('utf-8')

def object_to_dict(obj):
    """
    Converts an object into a dictionary, preserving its attributes.

    Args:
        obj: The object to be converted into a dictionary.

    Returns:
        A dictionary representation of the object, where the keys are the attribute names
        and the values are the attribute values.

    """
    result_dict = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, bytes):
            # If the attribute is binary, convert to Base64
            result_dict[key] = convert_binary_to_base64(value)
        else:
            # Otherwise, include the attribute as is
            result_dict[key] = value

    return result_dict



def save_data_to_mongo(inp, out, task_name, is_error=False):
    """
    Saves data for voice tasks into MongoDB with parameters input, output, and task name.

    Parameters:
    - inp: The input data to be saved.
    - out: The output data to be saved.
    - task_name: The name of the task.
    - is_error: A boolean flag indicating whether the data is an error or not. Default is False.

    Returns:
    - None if the data is saved successfully.
    - "Failed to save data." if there was an error while saving the data.
    - "Error: <error_message>" if there was an exception during the process.

    Note:
    - This function assumes that MongoDB is running on the local machine with the default port (27017).
    - The data is saved in the "saved_data" database.
    - If is_error is True, the data is saved in the "error_collection" collection. Otherwise, it is saved in the "saved_data" collection.
    """
    mongo_uri = "mongodb://localhost:27017/"
    database_name = "saved_data"
    
    if is_error:
        collection_name = "error_collection"
    else:
        collection_name = "saved_data"

    client = MongoClient(mongo_uri)
    database = client[database_name]
    collection = database[collection_name]

    try:
        inp = object_to_dict(inp)
        out = object_to_dict(out)
        data = {
                "input" : inp,
                "output" : out,
                "task_name" : task_name
                }
        result = collection.insert_one(data)
        if result.inserted_id:
            print(f"Data saved successfully in {collection_name} collection.")
        else:
            return print("Failed to save data.")
    except Exception as e:
        return print(f"Error: {str(e)}")
    finally:
        client.close()



@staticmethod
def preprocessing(audio: str, audioFileDirectoryPath: str) -> str:
    """
    Preprocesses the audio file by converting it to WAV format using ffmpeg.

    Args:
        audio (str): The path to the input audio file.
        audioFileDirectoryPath (str): The directory path where the processed audio file will be saved.

    Returns:
        str: The path to the processed audio file.

    Raises:
        None

    """
    if not os.path.exists(audioFileDirectoryPath):
        os.makedirs(audioFileDirectoryPath)

    audioFileName = f"{datetime.datetime.now().strftime('%H_%M_%S_%d_%m_%Y')}.wav"
    audioFilePath = os.path.join(audioFileDirectoryPath, audioFileName)

    #pre-processing via ffmpeg:
    print("Starting conversion to wav.........")
    os.system(f'ffmpeg -y -i "{audio}" -ar 16000 -ac 1 -c:a pcm_s16le "{audioFilePath}"')
    print('Conversion complete, file ready for transcription')

    return audioFilePath


@staticmethod

def split_and_save_audio(input_audio, chunk_size=15, output_dir="chunks"):
    """
    Splits an audio file into chunks of a specified duration and saves each chunk as a separate WAV file.

    Args:
        input_audio (str): Path to the input audio file.
        chunk_size (float, optional): Duration of each chunk in seconds. Defaults to 15.
        output_dir (str, optional): Directory to save the chunked audio files. Defaults to "chunks".

    Returns:
        List[str]: List of paths to the saved chunked audio files.
    """
    os.makedirs(output_dir, exist_ok=True)

    waveform, sample_rate = torchaudio.load(input_audio)

    chunk_samples = int(chunk_size * sample_rate)

    chunks = [waveform[:, i:i + chunk_samples] for i in range(0, waveform.size(1) - chunk_samples + 1, chunk_samples)]

    chunks = np.array([chunk.numpy() for chunk in chunks])

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(output_dir, f"chunk_{i}.wav")
        torchaudio.save(chunk_file_path, torch.from_numpy(chunk), sample_rate)
        chunk_files.append(chunk_file_path)

    return chunk_files