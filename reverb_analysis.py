from scipy.io.wavfile import read
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

def plot_audio_comparison(audio_path1, audio_path2):
    def load_and_stft(path):
        sr, audio = read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        f, t, Zxx = stft(audio, fs=sr, nperseg=1024)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        return f, t, magnitude, phase

    f1, t1, mag1, phase1 = load_and_stft(audio_path1)
    f2, t2, mag2, phase2 = load_and_stft(audio_path2)

    plt.figure(figsize=(14, 10))

    # Magnitude spectrograms
    plt.subplot(2, 2, 1)
    plt.pcolormesh(t1, f1, 20 * np.log10(mag1 + 1e-10), shading="gouraud")
    plt.title("STFT Magnitude (dB) - No reverb")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Magnitude (dB)")

    plt.subplot(2, 2, 2)
    plt.pcolormesh(t2, f2, 20 * np.log10(mag2 + 1e-10), shading="gouraud")
    plt.title("STFT Magnitude (dB) - With cathedral reverb")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Magnitude (dB)")

    # Phase difference (wrapped to [-π, π])
    # Phase spectrogram - No reverb
    plt.subplot(2, 2, 3)
    plt.pcolormesh(t1, f1, phase1, shading="gouraud", cmap="twilight")
    plt.title("STFT Phase - No reverb")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Phase (radians)")

    # Phase spectrogram - With cathedral reverb
    plt.subplot(2, 2, 4)
    plt.pcolormesh(t2, f2, phase2, shading="gouraud", cmap="twilight")
    plt.title("STFT Phase - With cathedral reverb")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Phase (radians)")

    plt.tight_layout()
    plt.show()
    plt.savefig("comparison_plot.pdf")

# Example usage:
plot_audio_comparison("audio/flute.wav", "audio/flute_out_cathedral.wav" )
