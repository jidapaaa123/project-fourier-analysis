import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

original_fourier_df = pd.read_csv('./Fourier_Space_Original.txt', names=['value'], sep=' ', header=None)
fourier_df = pd.read_csv('./Fourier_Space.txt', names=['value'], sep=' ', header=None)

# ====================================================================================================================================
# # Part 2 & 3: 
# # My laptop does not like it when it's too big
# # Trim down to first n points
# n = 500
# f = fourier_df.iloc[:n,:]    

# # Compute the Fast Fourier Transform (FFT)
# dt = 1/2000
# t = np.arange(0,dt*n,dt)
# fhat = np.fft.fft(f,n) # Compute the FFT

# PSD = fhat * np.conj(fhat) / n # Power spectrum (power per freq)
# freq = (1/(dt*n)) * np.arange(n) # Create x-axis of frequencies in Hz
# L = np.arange(0,n,dtype='int')

# # Use the PSD to filter out noise
# PSD_THRESHOLD = 50
# indices = PSD > PSD_THRESHOLD # Find all freqs with large power
# PSDclean = PSD * indices # Zero out all others

# NUM_SUBPLOTS = 3
# # 1. FFT magnitude
# plt.subplot(NUM_SUBPLOTS, 1, 1)
# plt.plot(freq[L], np.abs(fhat[L]), color='red')  # Magnitude of FFT
# plt.title('FFT Magnitude')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('|F(f)|')
# plt.grid(True)

# # 2. PSD
# plt.subplot(NUM_SUBPLOTS, 1, 2)
# plt.plot(freq[L], PSD[L], color='blue')
# plt.title('PSD')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude')
# plt.grid(True)

# # 3. Clean FFT based on PSD
# fhat = indices * fhat # Zero out small Fourier coeffs. in Y
# plt.subplot(NUM_SUBPLOTS, 1, 3)
# plt.plot(freq[L], np.abs(fhat[L]), color='red')  # Magnitude of FFT
# plt.title('Cleaned FFT Magnitude')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('|F(f)|')
# plt.grid(True)

# plt.show()


# # 4. Filtered signal
# ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal

# plt.subplot(NUM_SUBPLOTS, 1, 1)
# plt.plot(t, ffilt.real, color='green')  # Take real part after ifft
# plt.title('Filtered Time Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid(True)

# # Original signal as from Fourier_Space.txt
# plt.subplot(NUM_SUBPLOTS, 1, 2)
# plt.plot(t, f, color='green')  
# plt.title('Fourier_Space Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid(True)

# # Original signal as from Fourier_Space_Original.txt
# plt.subplot(NUM_SUBPLOTS, 1, 3)
# plt.plot(t, original_fourier_df.iloc[:n,:], color='green')  
# plt.title('Fourier_Space_Original Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid(True)

# plt.show()

# ====================================================================================================================================
# # Part 4: Process Data in 1-Second Windows
# windows = 5
# NUM_SUBPLOTS = windows

# for i in range(windows):
#     start_idx = i * 2000
#     end_idx = start_idx + 2000 # Make sure it's EXCLUSIVE

#     n = end_idx - start_idx # [0, 2000) -> 2000 points]
#     f = fourier_df.iloc[start_idx:end_idx,:]    

#     # Compute the Fast Fourier Transform (FFT)
#     dt = 1/2000
#     t = np.arange(0,dt*n,dt)
#     fhat = np.fft.fft(f,n) # Compute the FFT

#     PSD = fhat * np.conj(fhat) / n # Power spectrum (power per freq)
#     freq = (1/(dt*n)) * np.arange(n) # Create x-axis of frequencies in Hz
#     L = np.arange(0,n,dtype='int')

#     # Use the PSD to filter out noise
#     PSD_THRESHOLD = 50
#     indices = PSD > PSD_THRESHOLD # Find all freqs with large power
#     PSDclean = PSD * indices # Zero out all others
#     fhat = indices * fhat # Zero out small Fourier coeffs. in Y
#     ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal

#     plt.subplot(NUM_SUBPLOTS, 1, i+1)
#     plt.plot(freq[L], np.abs(fhat[L]), color='red')  # Magnitude of FFT
#     plt.title(f'Cleaned FFT Magnitude for {i}th-{i+1}th second')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('|F(f)|')
#     plt.grid(True)

# plt.show()

# ====================================================================================================================================
# Part 5: Spectrogram of Original Signal.
# It looks so strange and boring, I don't really know what this is I just tried the code in the notes
dt = 1/2000
n = 10000
x = original_fourier_df.values.flatten()[:n]
plt.figure(figsize=(10, 5))
plt.specgram(x, NFFT=128, Fs=1/dt, noverlap=120,cmap='jet')
plt.colorbar()
plt.title(f"Spectrogram of Original Signal (First {n} Points)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

# ====================================================================================================================================