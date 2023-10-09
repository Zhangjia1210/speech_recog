import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_analysis_with_librosa(input_audio_path):
    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(input_audio_path, sr=None)
    
    # Convert the audio data to frequency domain and get its amplitude in dB
    S = np.abs(librosa.stft(audio_data))
    amplitude_dB = librosa.amplitude_to_db(S, ref=np.max)
    
    # Get the frequencies corresponding to the rows in D
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    
    # Identify the maximum amplitude for each frequency
    max_amplitudes_per_frequency = np.max(amplitude_dB, axis=1)
    
    # Based on a threshold of -40dB, identify the Nyquist frequency
    nyquist_freq = np.max(frequencies[np.where(max_amplitudes_per_frequency > -40)]) * 2
    recommended_sampling_rate = max(44100, int(nyquist_freq))
    
    # Calculate the dynamic range in dB
    dB_dynamic_range = np.max(amplitude_dB) - np.min(amplitude_dB)
    recommended_bit_depth = np.ceil(dB_dynamic_range / 6)
    
    # Calculate the amplification coefficient
    peak_amplitude = np.max(np.abs(audio_data))
    maximum_value_for_bit_depth = (2 ** (recommended_bit_depth - 1)) - 1
    amplification_factor = maximum_value_for_bit_depth / peak_amplitude
    
    return recommended_sampling_rate, recommended_bit_depth, amplification_factor


# Test the function
audio_path = 'file_example_MP3_700KB.mp3'
rec_sample_rate, rec_bit_depth, ampl_factor = audio_analysis_with_librosa(audio_path)
print(f"Предложенная частота дискретизации: {rec_sample_rate} Гц")
print(f"Предложенная битовая глубина: {rec_bit_depth} бита")
print(f"Предельный коэффициент усиления: {ampl_factor:.3f}")