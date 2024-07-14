import numpy as np
from scipy.fft import fft
import pandas as pd
import mne
import mne_features
from scipy.stats import entropy
from scipy import stats

# Enabling cuda
mne.utils.set_config('MNE_USE_CUDA', 'true')


# Reading data
raw = mne.io.read_raw_edf("../../yasa_example_night_young.edf", preload=True)
data = raw.get_data()


# Fixing EEG, EOG and EMG Channels
yasa_ch_names = ['ROC-A1', 'LOC-A2', 'C3-A2', 'O2-A1', 'C4-A1', 'O1-A2',
       'EMG1-EMG2', 'Fp1-A2', 'Fp2-A1', 'F7-A2', 'F3-A2', 'FZ-A2',
       'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2',
       'PZ-A2', 'P4-A1', 'T6-A1', 'EKG-R-EKG-L']
yasa_ch_types = ['eog','eog','eeg','eeg','eeg','eeg','emg','eeg','eeg',
                 'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                 'eeg','eeg','eeg','eeg','misc']
info = mne.create_info(ch_names=yasa_ch_names, sfreq=200 ,ch_types=yasa_ch_types)
raw = mne.io.RawArray(data, info)
eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, emg=False, exclude=[])
emg_channels = mne.pick_types(raw.info, meg=False, eeg=False, emg=True, exclude=[])
eog_channels = mne.pick_types(raw.info, meg=False, eeg=False, emg=False,eog=True , exclude=[])


# Preprocessing
# Butterworth filter
iir_params = dict(order=10, ftype='butter')
raw.filter(l_freq=0.5, h_freq=49.5, method='iir',
            iir_params=iir_params, verbose=True, n_jobs=20)
# Notch filter
raw.notch_filter(freqs=50)
## EEG and EOG Channels Channels
# Low pass filter and high pass filter
raw.filter(l_freq=0.3, h_freq=30, fir_design='firwin', n_jobs=-1, picks=list(tuple(eeg_channels) + tuple(eog_channels)))

## EOG Channels
raw.filter(l_freq=10, h_freq=75, fir_design='firwin', n_jobs=-1, picks=emg_channels)


# Resampling
raw.resample(sfreq=100, n_jobs=-1)


# Divide in epochs
epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=True,reject_by_annotation=True)
# Define frequency bands
delta_band = (0.5, 4)
theta_band = (4, 8)
alpha_band = (8, 12)
beta_band = (12, 30)
    

def compute_relative_power(psd, freqs, band):
    band_power = np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])])
    total_power = np.sum(psd)
    relative_power = band_power / total_power
    return relative_power

def compute_band_power(psd, freqs, band):
    band_power = np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])])
    return band_power

def zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zcr = len(zero_crossings) / len(signal)
    return zcr

def spectral_entropy(psd):
    psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
    spec_ent = entropy(psd_norm, axis=-1)
    return spec_ent


# Define frequency bands
delta_band = (0.5, 4)
theta_band = (4, 8)
alpha_band = (8, 12)
beta_band = (12, 30)
    
# Feature extractions
def compute_relative_power(psd, freqs, band):
    band_power = np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])])
    total_power = np.sum(psd)
    relative_power = band_power / total_power
    return relative_power

def compute_band_power(psd, freqs, band):
    band_power = np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])])
    return band_power

def zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zcr = len(zero_crossings) / len(signal)
    return zcr

def spectral_entropy(psd):
    psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
    spec_ent = entropy(psd_norm, axis=-1)
    return spec_ent

def extract_features(raw):
    # Minimum value
    minV = np.min(raw)

    # Maximum value
    maxV = np.max(raw)
    
    # Std_Deviation
    std = np.std(raw)

    # Variance
    var = np.var(raw)

    # Mean
    mean = np.mean(raw)

    # Median
    median = np.median(raw)

    # Mode
    mode_ = stats.mode(raw)[0]

    # Skewness
    skew = stats.skew(raw)

    # Kurtosis
    kurtosis = stats.kurtosis(raw)

    # Hjorth Mobility
    hjorth_mobility = mne_features.univariate.compute_hjorth_mobility_spect(100, raw)

    # Hjorth Compexity
    hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(raw)

    # 75th percentile
    percentile = np.percentile(raw, 75)

    # Zero Cross Rate
    zcr = zero_crossing_rate(raw)

    # Relative spectral power
    psds, freqs = mne.time_frequency.psd_array_welch(raw, sfreq=100, fmin=0.5, fmax=30, n_fft=2048)
    rsp_delta = compute_relative_power(psds, freqs, delta_band)
    rsp_theta = compute_relative_power(psds, freqs, theta_band)
    rsp_alpha = compute_relative_power(psds, freqs, alpha_band)
    rsp_beta = compute_relative_power(psds, freqs, beta_band)

    # Compute band power
    band_pwr_delta = compute_band_power(psds, freqs, delta_band)
    band_pwr_theta = compute_band_power(psds, freqs, theta_band)
    band_pwr_alpha = compute_band_power(psds, freqs, alpha_band)
    band_pwr_beta = compute_band_power(psds, freqs, beta_band)

    # Spectral Entropy
    spec_entropy = spectral_entropy(psds)

    return [minV, maxV, std, var, mean, median, mode_, skew, kurtosis, hjorth_mobility, hjorth_complexity, percentile, zcr, rsp_delta, rsp_theta, 
            rsp_alpha, rsp_beta, band_pwr_delta, band_pwr_theta, band_pwr_alpha, band_pwr_beta, spec_entropy]

def extract_features_emg(raw):
    # Minimum value
    minV = np.min(raw)

    # Maximum value
    maxV = np.max(raw)
    
    # Std_Deviation
    std = np.std(raw)

    # Variance
    var = np.var(raw)

    # Mean
    mean = np.mean(raw)

    # Median
    median = np.median(raw)


    # Skewness
    skew = stats.skew(raw)

    # Kurtosis
    kurtosis = stats.kurtosis(raw)

    # Hjorth Mobility
    hjorth_mobility = mne_features.univariate.compute_hjorth_mobility_spect(100, raw)

    # Hjorth Compexity
    hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(raw)

    # Energy 
    energy= np.sum(raw**2) / len(raw)

    # Age
    age = 19

    # Approximate Entropy 
    # approximate_entropy  = mne_features.univariate.compute_sapp_entropy(raw)
    psds, freqs = mne.time_frequency.psd_array_welch(raw, sfreq=100, fmin=0.5, fmax=30, n_fft=2048)
    #Permutation Entropy
    permutation_entropy  = np.sum(psds * np.log2(psds)) / np.log2(len(psds))

    #Spectral Edge 
    # spectral_edge = mne_features.univariate.compute_spectral_edge_frequency(psds, freqs, edge=0.95)

    # Zero Cross Rate
    zcr = zero_crossing_rate(raw)

    # Spectral Entropy
    spec_entropy = spectral_entropy(psds)

    return [minV, maxV, std, var, mean, median, skew, kurtosis, hjorth_mobility, hjorth_complexity,zcr, spec_entropy,age,permutation_entropy]

def extract_features_eog(raw):
    # Minimum value
    minV = np.min(raw)


    # Std_Deviation
    std = np.std(raw)

    # Variance
    var = np.var(raw)

    # Mean
    mean = np.mean(raw)

    # Median
    median = np.median(raw)


    # Skewness
    skew = stats.skew(raw)

    # Kurtosis
    kurtosis = stats.kurtosis(raw)

    # Hjorth Mobility
    hjorth_mobility = mne_features.univariate.compute_hjorth_mobility_spect(100, raw)

    # Hjorth Compexity
    hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(raw)

   

    # Approximate Entropy 
    # approximate_entropy  = mne_features.univariate.compute_sapp_entropy(raw)
    psds, freqs = mne.time_frequency.psd_array_welch(raw, sfreq=100, fmin=0.5, fmax=30, n_fft=2048)
    #Permutation Entropy
    permutation_entropy  = np.sum(psds * np.log2(psds)) / np.log2(len(psds))

    #Spectral Edge 
    # spectral_edge = mne_features.univariate.compute_spectral_edge_frequency(psds, freqs, edge=0.95)

    # Zero Cross Rate
    zcr = zero_crossing_rate(raw)

    # Spectral Entropy
    spec_entropy = spectral_entropy(psds)

    return [minV, std, var, mean, median, skew, kurtosis, hjorth_mobility, hjorth_complexity,zcr, spec_entropy,permutation_entropy]

features = []

# for each eeg channel 
features_list_eeg = []
for epoch in epochs:
    epoch_features = []
    for ch in eeg_channels:
        epoch_features.append(extract_features(epoch[ch]))
    features_list_eeg.append(epoch_features)
    
#for each emg channels
features_list_emg = []
for epoch in epochs:
    epoch_features = []
    for ch in emg_channels:
        epoch_features.append(extract_features_emg(epoch[ch]))
    features_list_emg.append(epoch_features)


# # for each eog channels

features_list_eog = []
for epoch in epochs:
    epoch_features = []
    for ch in eog_channels:
        epoch_features.append(extract_features_eog(epoch[ch]))
    features_list_eog.append(epoch_features)

# Saving arrays
np.save('./eeg/yasadata.npy', features_list_eeg)
np.save('./emg/yasadata.npy', features_list_emg)
np.save('./eog/yasadata.npy', features_list_eog)

