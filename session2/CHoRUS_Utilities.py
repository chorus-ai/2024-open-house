# -*- coding: utf-8 -*-
"""
morteza.zabihi@gmail.com

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.fft import fft
from scipy.stats import entropy, skew, kurtosis
from scipy.interpolate import interp1d
import pandas as pd
from collections import Counter
import librosa
from sklearn.model_selection import KFold, StratifiedKFold
import mne
import pywt
import statsmodels.api as sm

# =============================================================================
# ==============================Utility Functions==============================   
# =============================================================================   


def r_detection(ecg_signal, fs, wd=5, plot=False, preprocess=False):
    """
    Parameters
    ----------
    ecg_signal : 1 channel ECG
    fs : Sampling frequency
    wd: in seconds
    plot : The option to plot, not recommended if the signal is long.
    The default is False.

    Returns
    -------
    peaks_all_array : the numpy array of contains the location of Rs

    """
    peaks_all_array = []
    # ---------------------------------------------------------------------
    # preprocessing -------------------------------------------------------
    if preprocess:
        level = 7
        Wtype = 'db4'
        
        coeffs  = pywt.wavedec(ecg_signal, Wtype, level=level)
        cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs        
        
        Mapp = np.zeros_like(cA7)
        McD2 = np.zeros_like(cD2)
        McD1 = np.zeros_like(cD1)
        
        Mc = [Mapp, cD7, cD6, cD5, cD4, cD3, McD2, McD1]
        recD = pywt.waverec(Mc, Wtype)
    else:
        recD = ecg_signal
    # thr detection -------------------------------------------------------
    thrs = []
    
    l = wd*fs
    # Iterate over windows
    for i in range(0, len(recD), l):
        window = recD[i:i+l]  # Extract windowed data
        if np.var(window) > np.finfo(np.float32).eps:
            thr = np.percentile(np.abs(window), 98)
            thrs.append(thr)
    
    thr = np.percentile(thrs, 1)
    
    # peak detection ------------------------------------------------------
    peaks_all = []
    min_peak_distance = int(0.4*fs)
    
    # Iterate over windows
    for i in range(0, len(recD), l):
        window = recD[i:i+l]  # Extract windowed data
        if np.var(window) > 0.000001:
            peaks, _ = find_peaks(np.abs(window), height=thr, distance=min_peak_distance)
            peaks_all.append(peaks + i)
            
    peaks_all_array = np.concatenate(peaks_all)
    # post processing -----------------------------------------------------
    for iter1 in range(1, len(peaks_all_array)-1):
        peak = peaks_all_array[iter1]
        
        llim = peak - int(0.125*fs)
        hlim = peak + int(0.125*fs)
        if llim >0 & hlim <len(ecg_signal):
            if llim>peaks_all_array[iter1-1] & hlim<peaks_all_array[iter1+1]:
                window = ecg_signal[llim:hlim]
                window = np.abs(window)
                
                peak_new = np.argmax(window)
                peak_new += llim
                peaks_all_array[iter1] = peak_new
    # post processing 1 (if the peaks inteval is more than 1.6s)-----------
    intervals = np.diff(peaks_all_array)
    missedlocs = np.where(intervals > fs*1.6)[0]
    if len(missedlocs) > 0:
        peaks_new = []
        for iter1 in range(len(missedlocs)):
            llim = peaks_all_array[missedlocs[iter1]]
            hlim = peaks_all_array[missedlocs[iter1]+1]
            window = ecg_signal[llim:hlim]
            window = np.abs(window)
            peak_new = np.argmax(window)
            peak_new += llim
            peaks_new.append(peak_new)
        
        peaks_new = np.array(peaks_new)
        peaks_all_array = np.append(peaks_all_array, peaks_new)
        peaks_all_array = np.sort(peaks_all_array)
    # post processing 2 (if the peaks are closer than 0.099s)--------------
    intervals = np.diff(peaks_all_array)
    tooclose = np.where(intervals < 0.099*fs)[0]
    peaks_all_array = np.delete(peaks_all_array, tooclose)
    # post processing -----------------------------------------------------
    for iter1 in range(1, len(peaks_all_array)-1):
        peak = peaks_all_array[iter1]
        
        llim = peak - int(0.125*fs)
        hlim = peak + int(0.125*fs)
        if llim >0 & hlim <len(ecg_signal):
            if llim>peaks_all_array[iter1-1] & hlim<peaks_all_array[iter1+1]:
                window = ecg_signal[llim:hlim]
                window = np.abs(window)
                
                peak_new = np.argmax(window)
                peak_new += llim
                peaks_all_array[iter1] = peak_new
    # plot ----------------------------------------------------------------
    if plot:
        num_samples = len(ecg_signal)
        time_vector = np.arange(num_samples) / fs
        
        plt.figure()
        plt.plot(time_vector, ecg_signal)
        plt.plot(time_vector[peaks_all_array], ecg_signal[peaks_all_array], 'ro')
        plt.title('Signal with Peaks')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
        
    return peaks_all_array, recD
# =============================================================================   


def kl_divergence_score(y_true_probs, y_pred_probs, epsilon=1e-16):
    kl_scores = []

    # ---------------------------------------------------------------------
    for true_label, pred_probs in zip(y_true_probs, y_pred_probs):
        # ---------------------------------------------------------------------
        true_probs = np.zeros_like(pred_probs)
        true_probs[true_label] = 1
        # ---------------------------------------------------------------------
        pred_probs = np.clip(pred_probs, epsilon, 1 - epsilon)
        # ---------------------------------------------------------------------
        kl_score = entropy(true_probs, pred_probs)
        kl_scores.append(kl_score)
        # ---------------------------------------------------------------------
        
    # Average KL divergence over all samples
    kl_score = np.mean(kl_scores)
    return kl_score
# =============================================================================   


def cross_validation(labels, n_folds=5, random_state=42, method="kfold"):
    
    if method == "kfold":
        # Method 1: Randomly split into 5 exclusive sets
        train_folds, test_folds = [], []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for train_index, test_index in kf.split(labels):
            train_folds.append(train_index)
            test_folds.append(test_index)
    # -------------------------------------------------------------------------
    elif method == "skfold":
        # Method 2: stratified split into 5 exclusive sets
        train_folds, test_folds = [], []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for train_index, test_index in skf.split(labels, labels):
            train_folds.append(train_index)
            test_folds.append(test_index)
    # -------------------------------------------------------------------------
    return train_folds, test_folds
# ============================================================================= 


def balance_classes_random_upsampling(X, y):
    # Find the class with the maximum number of samples
    max_class_samples = max(Counter(y).values())

    # Create a mask for each class to randomly upsample
    masks = [np.random.choice(np.where(y == label)[0], max_class_samples, replace=True) for label in np.unique(y)]

    # Combine the masks to get the final indices for upsampling
    indices_to_duplicate = np.concatenate(masks)

    # Use the indices to duplicate samples and get the balanced subset of data
    X_balanced = X[indices_to_duplicate, :]
    y_balanced = y[indices_to_duplicate]

    return X_balanced, y_balanced, indices_to_duplicate
# =============================================================================  


def balance_classes_random_downsampling(X, y):
    # Find the class with the minimum number of samples
    min_class_samples = min(Counter(y).values())

    # Create a mask for each class to randomly downsample
    masks = [np.random.choice(np.where(y == label)[0], min_class_samples, replace=False) for label in np.unique(y)]

    # Combine the masks to get the final indices for downsampling
    indices_to_keep = np.concatenate(masks)

    # Use the indices to get the balanced subset of data
    X_balanced = X[indices_to_keep, :]
    y_balanced = y[indices_to_keep]

    return X_balanced, y_balanced, indices_to_keep
# =============================================================================   


def feature_extraction(x, fs):
    R_i2, x = r_detection(x, fs, wd=5, plot=False, preprocess=True)
    # -------------------------------------------------------------------------
    features_beat, beats = beat_features(x, R_i2, fs)
    features_beat_1 = beat_features_v1(beats)
    adfeatures = advanced_features(x, fs)
    shfeatures = sh_vs_nshself(x, fs)
    hrvfeatures = hrv_features(R_i2, fs)
    # -------------------------------------------------------------------------
    all_features = np.hstack((features_beat, features_beat_1, adfeatures,
                              shfeatures, hrvfeatures))
    return all_features
# =============================================================================   


def cosine_similarity(row1, row2):
    dot_product = np.dot(row1, row2)
    norm_row1 = np.linalg.norm(row1)
    norm_row2 = np.linalg.norm(row2)
    similarity = dot_product / (norm_row1 * norm_row2)
    return similarity
# =============================================================================   


def beat_features_v1(beats):
    
    if len(beats)>3:
        beats_1 = np.zeros((len(beats), 201))
        for iter1 in range(len(beats)):
            beats_1[iter1, :] = interpolate_to_201_samples(beats[iter1])
        
        num_rows = beats_1.shape[0]
        cosine_similarities = []
        average_period_1 = []
        for i in range(num_rows):
            for j in range(i+1, num_rows):
                similarity = cosine_similarity(beats_1[i, :], beats_1[j, :])
                cosine_similarities.append(similarity)
                
                autocorr = np.correlate(beats_1[i, :], beats_1[j, :], mode='full')[201 // 2:]
                autocorr = autocorr / np.max(autocorr)
                peaks, _ = find_peaks(autocorr)
                average_period_1.append(np.mean(np.diff(peaks)) if len(peaks) > 3 else 0)
                
        cs_m = np.nanmean(cosine_similarities)
        cs_v = np.nanvar(cosine_similarities)
        
        corr_m = np.nanmean(average_period_1)
        corr_v = np.nanvar(average_period_1)
        
        covariance_matrix = np.cov(beats_1)
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        eigstats = np.percentile(eigenvalues, [0.25, 0.5, 0.75, 0.99])
        # ---------------------------------------------------------------------
        mbeats = np.sum(beats_1, axis=0)
        
        zero_crossings = np.where(np.diff(np.signbit(mbeats)))[0]
        if len(zero_crossings)>4:
            difzc = np.diff(zero_crossings)
            q75, q25 = np.percentile(difzc, [75 ,25])
            iqr_1625 = q75 - q25
        else:
            iqr_1625 = 0
        # ---------------------------------------------------------------------
        gradient_1 = np.gradient(mbeats)
        gradient_2 = np.gradient(gradient_1)
        ha = np.var(mbeats)
        hm = np.sqrt(np.var(gradient_1) / ha)
        hm1 = np.sqrt(np.var(gradient_2) / np.var(gradient_1))
        hc = hm1 / hm
        # ---------------------------------------------------------------------
        _, counts = np.unique(mbeats, return_counts=True)
        total_count = len(mbeats)
        frequencies = counts / total_count
        ent = -np.sum(frequencies * np.log2(frequencies))
        # ---------------------------------------------------------------------
        features_beats_1 = np.array([cs_m, cs_v, corr_m, corr_v, iqr_1625, ha, hm, hc, ent])
        features_beats_1 = np.hstack((features_beats_1, eigstats))
    else:
        features_beats_1 = np.zeros((13, ))
    
    return features_beats_1
# =============================================================================   
  

def beat_features(xd1, R_i2, fs):
    PAR_1, NAR_1, md_1, vd_1, PP_1 = [], [], [], [], []
    PAR_2, NAR_2, md_2, vd_2, PP_2 = [], [], [], [], []
    PAR_3, NAR_3, md_3, vd_3, PP_3 = [], [], [], [], []
    PAR_4, NAR_4, md_4, vd_4, PP_4 = [], [], [], [], []
    
    beats = []
    if len(R_i2)>3:
        
        for i1 in range(1, len(R_i2)-1):
            Endd = (R_i2[i1+1] - R_i2[i1]) // 2
            Startt = (R_i2[i1] - R_i2[i1-1]) // 2
            
            temp = xd1[R_i2[i1]-Startt : R_i2[i1]+Endd]            
            temp = (temp - np.mean(temp))/np.std(temp)
            beats.append(temp)
            
            R_new = R_i2[i1] - (R_i2[i1] - Startt) + 1
            # ---------------------------------------------------------------------
            start_p = int(R_new - 0.2*fs)
            end_p = int(R_new - 0.05*fs)
            
            start_QRS = int(R_new - 0.05*fs)
            end_QRS = int(R_new + 0.05*fs)
            
            end_QT = int(start_QRS + 0.42*fs)
            
            start_T = int(end_QRS +  0.05*fs)
            # ---------------------------------------------------------------------
            if start_p>=0 and end_p<len(temp):
                pwave = temp[start_p:end_p]
                
                PAR = np.sum(0.5 * (pwave + np.abs(pwave))) / len(pwave)
                NAR = np.sum(0.5 * (pwave - np.abs(pwave))) / len(pwave)
                md = np.mean(np.diff(pwave)) / len(pwave)
                vd = np.var(np.diff(pwave)) / len(pwave)
                PP = np.max(pwave) - np.min(pwave)
            
            else:
                PAR = 0
                NAR = 0
                md = 0
                vd = 0
                PP = 0
            
            PAR_1.append(PAR)
            NAR_1.append(NAR)
            md_1.append(md)
            vd_1.append(vd)
            PP_1.append(PP)
            # ---------------------------------------------------------------------
            if start_QRS>=0 and end_QRS<len(temp):
                qrswave = temp[start_QRS:end_QRS]
                
                PAR = np.sum(0.5 * (qrswave + np.abs(qrswave))) / len(qrswave)
                NAR = np.sum(0.5 * (qrswave - np.abs(qrswave))) / len(qrswave)
                md = np.mean(np.diff(qrswave)) / len(qrswave)
                vd = np.var(np.diff(qrswave)) / len(qrswave)
                PP = np.max(qrswave) - np.min(qrswave)
            else:
                PAR = 0
                NAR = 0
                md = 0
                vd = 0
                PP = 0
                
            PAR_2.append(PAR)
            NAR_2.append(NAR)
            md_2.append(md)
            vd_2.append(vd)
            PP_2.append(PP)
            # ---------------------------------------------------------------------
            if start_QRS>=0 and end_QT<len(temp):
                qtwave = temp[start_QRS:end_QT]
                
                PAR = np.sum(0.5 * (qtwave + np.abs(qtwave))) / len(qtwave)
                NAR = np.sum(0.5 * (qtwave - np.abs(qtwave))) / len(qtwave)
                md = np.mean(np.diff(qtwave)) / len(qtwave)
                vd = np.var(np.diff(qtwave)) / len(qtwave)
                PP = np.max(qtwave) - np.min(qtwave)
            else:
                PAR = 0
                NAR = 0
                md = 0
                vd = 0
                PP = 0
                
            PAR_3.append(PAR)
            NAR_3.append(NAR)
            md_3.append(md)
            vd_3.append(vd)
            PP_3.append(PP)
            # ---------------------------------------------------------------------   
            if end_QRS>=0 and start_T<len(temp):
                stwave = temp[end_QRS:start_T]
                            
                PAR = np.sum(0.5 * (stwave + np.abs(stwave))) / len(stwave)
                NAR = np.sum(0.5 * (stwave - np.abs(stwave))) / len(stwave)
                md = np.mean(np.diff(stwave)) / len(stwave)
                vd = np.var(np.diff(stwave)) / len(stwave)
                PP = np.max(stwave) - np.min(stwave)
            else:
                PAR = 0
                NAR = 0
                md = 0
                vd = 0
                PP = 0
            
            PAR_4.append(PAR)
            NAR_4.append(NAR)
            md_4.append(md)
            vd_4.append(vd)
            PP_4.append(PP)
            # ---------------------------------------------------------------------
        f1 = np.percentile(PAR_1, [25, 75])
        f2 = np.percentile(PAR_2, [25, 75])
        f3 = np.percentile(PAR_3, [25, 75])
        f4 = np.percentile(PAR_4, [25, 75])
        
        f5 = np.percentile(NAR_1, [25, 75])
        f6 = np.percentile(NAR_2, [25, 75])
        f7 = np.percentile(NAR_3, [25, 75])
        f8 = np.percentile(NAR_4, [25, 75])
        
        f9 = np.percentile(md_1,  [25, 75])
        f10 = np.percentile(md_2, [25, 75])
        f11 = np.percentile(md_3, [25, 75])
        f12 = np.percentile(md_4, [25, 75])
        
        f13 = np.percentile(vd_1, [25, 75])
        f14 = np.percentile(vd_2, [25, 75])
        f15 = np.percentile(vd_3, [25, 75])
        f16 = np.percentile(vd_4, [25, 75])
        
        f17 = np.percentile(PP_1, [25, 75])
        f18 = np.percentile(PP_2, [25, 75])
        f19 = np.percentile(PP_3, [25, 75])
        f20 = np.percentile(PP_4, [25, 75])
        
        f1 = f1[1] - f1[0]
        f2 = f2[1] - f2[0]
        f3 = f3[1] - f3[0]
        f4 = f4[1] - f4[0]
        f5 = f5[1] - f5[0]
        f6 = f6[1] - f6[0]
        f7 = f7[1] - f7[0]
        f8 = f8[1] - f8[0]
        f9 = f9[1] - f9[0]
        f10 = f10[1] - f10[0]
        f11 = f11[1] - f11[0]
        f12 = f12[1] - f12[0]
        f13 = f13[1] - f13[0]
        f14 = f14[1] - f14[0]
        f15 = f15[1] - f15[0]
        f16 = f16[1] - f16[0]
        f17 = f17[1] - f17[0]
        f18 = f18[1] - f18[0]
        f19 = f19[1] - f19[0]
        f20 = f20[1] - f20[0]
    else:
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        f5 = 0
        f6 = 0
        f7 = 0
        f8 = 0
        f9 = 0
        f10 = 0
        f11 = 0
        f12 = 0
        f13 = 0
        f14 = 0
        f15 = 0
        f16 = 0
        f17 = 0
        f18 = 0
        f19 = 0
        f20 = 0
        
    
    features_beat = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
                              f12, f13, f14, f15, f16, f17, f18, f19, f20])
    return features_beat, beats
# =============================================================================   
  

def advanced_features(row1, fs):
    
    l = len(np.squeeze(row1))
    # -------------------------------------------------------------------------
    coeffs = pywt.wavedec(row1, 'db4', level=5)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    # -------------------------------------------------------------------------
    en_cD5 = np.sum(np.power(cD5, 2))
    en_cD4 = np.sum(np.power(cD4, 2))
    en_cD3 = np.sum(np.power(cD3, 2))
    en_cD2 = np.sum(np.power(cD2, 2))
    en_cD1 = np.sum(np.power(cD1, 2))
    en_sum = en_cD1 + en_cD2 + en_cD3 + en_cD4 + en_cD5
    
    en_cD5 = en_cD5/en_sum
    en_cD4 = en_cD4/en_sum
    en_cD3 = en_cD3/en_sum
    en_cD2 = en_cD2/en_sum
    en_cD1 = en_cD1/en_sum
    
    TWE = en_cD5*np.log(en_cD5) + en_cD4*np.log(en_cD4) +\
        en_cD3*np.log(en_cD3) + en_cD2*np.log(en_cD2) + en_cD1*np.log(en_cD1)
    TWE = TWE/l
    # -------------------------------------------------------------------------  
    zero_crossings = np.where(np.diff(np.signbit(row1)))[0]
    if len(zero_crossings)>4:
        difzc = np.diff(zero_crossings)
        q75, q25 = np.percentile(difzc, [75 ,25])
        iqr_1625 = q75 - q25
    else:
        iqr_1625 = 0
    # -------------------------------------------------------------------------
    gradient_1 = np.gradient(row1)
    gradient_2 = np.gradient(gradient_1)
    ha = np.var(row1)
    hm = np.sqrt(np.var(gradient_1) / ha)
    hm1 = np.sqrt(np.var(gradient_2) / np.var(gradient_1))
    hc = hm1 / hm
    
    autocorr = np.correlate(row1, row1, mode='full')[row1.size // 2:]
    autocorr = autocorr / np.max(autocorr)
    peaks, _ = find_peaks(autocorr)
    average_period_1 = np.mean(np.diff(peaks)) if len(peaks) > 3 else 0
    # -------------------------------------------------------------------------
    f, psd = signal.welch(row1, fs, window='hamming', nperseg=2*fs)
    # -------------------------------------------------------------------------
    psd_norm = psd / psd.sum()
    se = -(psd_norm * np.log2(psd_norm)).sum()
    # -------------------------------------------------------------------------
    ret = spectral_edge_frequency(psd, f, edges=[0.5, 0.7, 0.8, 0.9, 0.95]) # 5
    # -------------------------------------------------------------------------
    freq_band_low = (3, 10)
    freq_band_med = (10, 30)
    freq_band_high = (30, 45)
    # Get frequency band
    freq_band_low_index = np.logical_and(f >= freq_band_low[0], f < freq_band_low[1])
    freq_band_med_index = np.logical_and(f >= freq_band_med[0], f < freq_band_med[1])
    freq_band_high_index = np.logical_and(f >= freq_band_high[0], f < freq_band_high[1])

    # Calculate maximum power
    max_power_low = np.max(psd[freq_band_low_index])
    max_power_med = np.max(psd[freq_band_med_index])
    max_power_high = np.max(psd[freq_band_high_index])

    # Calculate average power
    mean_power_low = np.trapz(y=psd[freq_band_low_index], x=f[freq_band_low_index])
    mean_power_med = np.trapz(y=psd[freq_band_med_index], x=f[freq_band_med_index])
    mean_power_high = np.trapz(y=psd[freq_band_high_index], x=f[freq_band_high_index])

    # Calculate max/mean power ratio
    f1_psd = max_power_low / mean_power_low
    f2_psd = max_power_med / mean_power_med
    f3_psd = max_power_high / mean_power_high
    # -------------------------------------------------------------------------    
    rho, sigma2 = sm.regression.linear_model.burg(row1, order=10)
    rho = rho/l
    # -------------------------------------------------------------------------
    spectrogram1, frequencies_r, time_r = spectrogram(row1, fs)
    # -------------------------------------------------------------------------
    fr = np.where((frequencies_r>0) & (frequencies_r<=40))[0]
    temp = np.squeeze(spectrogram1[fr, :])
    # -------------------------------------------------------------------------
    indstft = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
    fmax_stft = frequencies_r[fr[indstft[0]]]
    # -------------------------------------------------------------------------
    var_stft = np.var(temp)
    # -------------------------------------------------------------------------
    analytic_signal = hilbert(np.sum(temp, axis=0))
    # -------------------------------------------------------------------------
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi))
    mins = np.mean(instantaneous_frequency)
    sins = np.std(instantaneous_frequency)
    # ------------------------------------------------------------------------- 
    sa1 = seasonal_autocorrelation(row1, int(0.3*fs))
    sa2 = seasonal_autocorrelation(row1, int(0.8*fs))
    sa3 = seasonal_autocorrelation(row1, int(1.5*fs))
    # -------------------------------------------------------------------------
    _, counts = np.unique(row1, return_counts=True)
    frequencies = counts / l
    ent = -np.sum(frequencies * np.log2(frequencies))
    # -------------------------------------------------------------------------
    nzc = np.diff(np.signbit(row1)).sum()
    pfd = np.log10(l) / (np.log10(l) + np.log10(l / (l + 0.4 * nzc)))
    # -------------------------------------------------------------------------
    dists = np.abs(np.diff(row1))
    ll = dists.sum()
    ln = np.log10(ll / dists.mean())
    aux_d = row1 - np.take(row1, indices=[0])
    d = np.max(np.abs(aux_d))
    kfd = np.squeeze(ln / (ln + np.log10(d / ll)))
    # -------------------------------------------------------------------------
    coeffs, _ = pywt.cwt(row1, np.arange(1, 128), 'mexh')
    
    _, counts = np.unique(coeffs.flatten(), return_counts=True)
    total_count = len(coeffs.flatten())
    prob = counts / total_count
    prob = np.reshape(prob, (coeffs.shape[0], coeffs.shape[1]))
            
    rows, cols = coeffs.shape
    row_indices = np.arange(rows)[:, np.newaxis]  # Create an array of row indices
    col_indices = np.arange(cols)  # Create an array of column indices
    diff_matrix = np.abs(row_indices - col_indices)
    
    prob = np.divide(prob, diff_matrix+1)
    homogeneity = np.sum(prob) / total_count
    # -------------------------------------------------------------------------
    features = np.array([TWE, iqr_1625, ha, hm, hc, np.var(cD5), np.var(cD4),
                         np.var(cD3), average_period_1, se, fmax_stft, var_stft,
                         mins, sins, f1_psd, f2_psd, f3_psd, sa1, sa2, sa3,
                         ent, pfd, kfd, homogeneity])
    features = np.hstack((features, ret, rho))
    return features
# =============================================================================   


def sh_vs_nshself(row, fs):
    lowcut = 6.5
    highcut = 30
    order = 5
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, row)
    
    # Normalize to amplitude one
    normalized_ecg = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    f1 = np.max(normalized_ecg) - np.min(normalized_ecg)
    # ---------------------------------------------------------------------
    diff_ecg = np.diff(row)
    # Square the first-difference and normalize to amplitude one
    xd = (diff_ecg ** 2) / np.max(diff_ecg ** 2)
    
    # Compute the proportion of time xd is below the threshold (ThS)
    ThS = 0.1
    f2 = np.mean(xd < ThS)
    # ---------------------------------------------------------------------
    # Calculate the 1024-point FFT of the ECG segment
    power_proportion = 0.2
    fft_result = fft(row, 1024)
    # Calculate the power spectrum
    power_spectrum = np.abs(fft_result) ** 2

    # Normalize the power spectrum to unit area under the curve
    normalized_spectrum = power_spectrum / np.sum(power_spectrum)

    # Calculate the cumulative sum of the normalized spectrum
    cumulative_spectrum = np.cumsum(normalized_spectrum)

    # Find the frequency above which the given proportion of power is contained
    fH_index = np.argmax(cumulative_spectrum >= power_proportion)
    fH = fH_index / 1024

    # Find the frequency below which the given proportion of power is contained
    fL_index = np.argmax(cumulative_spectrum >= (1 - power_proportion))
    fL = fL_index / 1024

    # Calculate the bandwidth
    f3 = fH - fL
    
    fshnsh = np.hstack((f1, f2, f3))
    # ---------------------------------------------------------------------
    return fshnsh
# =============================================================================   


def hrv_features(R_i2, fs):
    # ---------------------------------------------------------------------
    if len(R_i2) > 3:
        diff_nni = np.diff(R_i2) # in sample
        diff_nni = diff_nni / fs # in seconds
        diff_nni = diff_nni * 1000 # in millisecond 
        
        NNx = sum(np.abs(diff_nni) > 50)
        # ---------------------------------------------------------------------
        length_int = len(R_i2)
        diff_nni = np.diff(R_i2)
        nni_50 = sum(np.abs(diff_nni) > 50)
        pNNx = 100 * nni_50 / length_int
        # ---------------------------------------------------------------------
        diff_nn_intervals = np.diff(R_i2)
        SD1 =  np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
        # ---------------------------------------------------------------------
        SD2 =  np.sqrt(2 * np.std(R_i2, ddof=1) ** 2 - 0.5 * np.std(\
                       diff_nn_intervals, ddof=1) ** 2)
        # ---------------------------------------------------------------------
        ratio_sd2_sd1 = SD2 / (SD1 + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------       
        L = 4 * SD1
        T = 4 * SD2
        CSI = L/T
        # ---------------------------------------------------------------------
        CVI = np.log10((L * T) + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------     
        modifiedCVI =  L ** 2 / T
        # ---------------------------------------------------------------------
        ff21 = np.mean(diff_nn_intervals)
        ff22 = np.std(diff_nn_intervals)
        f1 = np.sqrt(np.mean(diff_nn_intervals ** 2))
        f2 = ff22 / ff21
        # ---------------------------------------------------------------------
        sk_RR = skew(R_i2)
        kurt_RR = kurtosis(R_i2)
        # ---------------------------------------------------------------------
        r = 0.2*np.std(diff_nni)
        sampen_value = calculate_sampen(diff_nni, 2, r)
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        XXYY = np.diff(R_i2) / fs
        XX1 = XXYY
        XX2 = (np.mean(XXYY) - XXYY)**2
        p_coefficients = np.polyfit(XX1, XX2, 2)
        # ---------------------------------------------------------------------
        Ax = np.min(XXYY)
        Ay = np.abs(np.mean(XXYY) - np.min(XXYY))
        Bx = np.max(XXYY)
        By = np.abs(np.mean(XXYY) - np.max(XXYY))
        Cy = np.abs(np.mean(XXYY) - XXYY)
        I = np.argmin(Cy)
        Cy = Cy[I]
        Cx = XXYY[I]
        
        # f1 = (By - Ay) / (Bx - Ax)
        a = np.sqrt((Bx - Cx)**2 + (By - Cy)**2)
        b = np.sqrt((Ax - Cx)**2 + (Ay - Cy)**2)
        c = np.sqrt((Bx - Ax)**2 + (By - Ay)**2)
        f5 = a + b + c
        f6 = 0.5 * (((Ax - Cx) * (By - Ay)) - ((Ax - Bx) * (Cy - Ay)))
        f7 = (4 * np.sqrt(3) * f6) / ((a**2) + (b**2) + (c**2))
        # ---------------------------------------------------------------------
        XX11 = XXYY[:-1]
        YY22 = XXYY[1:]
        DISTI = (YY22 - XX11) / np.sqrt(2)
        
        indxU = np.where(DISTI == 0)[0]
        indxO = np.where(DISTI > 0)[0]
        indxD = np.where(DISTI < 0)[0]
        
        # f8 = len(indxU)
        # f9 = len(indxO)
        # f10 = len(indxD)
        
        UU = UO = UD = OU = OO = OD = DU = DO = DD = 0
        for Pi in range(len(YY22) - 1):
            if Pi in indxU and (Pi + 1) in indxU:
                UU += 1
            if Pi in indxU and (Pi + 1) in indxO:
                UO += 1
            if Pi in indxU and (Pi + 1) in indxD:
                UD += 1
            if Pi in indxO and (Pi + 1) in indxU:
                OU += 1
            if Pi in indxO and (Pi + 1) in indxO:
                OO += 1
            if Pi in indxO and (Pi + 1) in indxD:
                OD += 1
            if Pi in indxD and (Pi + 1) in indxU:
                DU += 1
            if Pi in indxD and (Pi + 1) in indxO:
                DO += 1
            if Pi in indxD and (Pi + 1) in indxD:
                DD += 1
        # ---------------------------------------------------------------------
        feature = [p_coefficients[2], f5, f6, f7, OU, OO, OD, DU, DD]
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        feature_HRV_1 = np.array([NNx, pNNx, SD1, SD2, ratio_sd2_sd1, CSI, CVI, 
                         modifiedCVI, f1, f2, sk_RR, kurt_RR, sampen_value])
        feature_HRV_1 = np.hstack((feature_HRV_1, feature))
    else:
        feature_HRV_1 = np.zeros((22, ))
    return feature_HRV_1    
# =============================================================================   


def calculate_sampen(rr_intervals, m, r):
    """
    Calculate Sample Entropy (SampEn).

    Args:
    - rr_intervals (list): List of RR intervals.
    - m (int): Template length.
    - r (float): Matching tolerance.

    Returns:
    - sampen (float): Sample Entropy value.
    """
    def _maxdist(xm_i, xm_j):
        """
        Calculate the maximum distance between two templates.
        """
        return max([abs(xm_i[k] - xm_j[k]) for k in range(len(xm_i))])

    def _phi(m):
        """
        Calculate phi(m), the number of matches of length m.
        """
        N = len(rr_intervals)
        count = 0
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                if i != j:
                    if _maxdist(rr_intervals[i:i+m], rr_intervals[j:j+m]) <= r:
                        count += 1
        return count

    def _sampen(m):
        """
        Calculate Sample Entropy for a given template length m.
        """
        N = len(rr_intervals)
        return -np.log( (_phi(m+1)+ np.finfo(float).eps) / (_phi(m) + np.finfo(float).eps))

    return _sampen(m)
# =============================================================================   


def read_waveform(record):
    # Read waveform samples (input is in WFDB-MAT format)
    mat_data = scipy.io.loadmat(record + ".mat")
    samples = mat_data['val']
    return samples
# =============================================================================   


def load_annotations(annot_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(annot_path, header=None)
    # Map values in the second column to numerical representations
    mapping = {'N': 0, 'A': 1, 'O': 2, '~': 3}
    df[2] = df[1].map(mapping)    
    return df
# =============================================================================   


def interpolate_to_201_samples(time_series):
    middle_index = len(time_series) // 2
    # Extract part of the time series before and after the middle point
    before_middle = time_series[:middle_index ]  # Include the middle point
    after_middle = time_series[middle_index+1:]

    # Create linear interpolation functions for before and after the middle point
    f_before = interp1d(np.arange(len(before_middle)), before_middle, kind='linear', fill_value="extrapolate")
    f_after = interp1d(np.arange(len(after_middle)), after_middle, kind='linear', fill_value="extrapolate")

    # Generate 100 additional samples before and after the middle point
    interpolated_before = f_before(np.linspace(0, len(before_middle) - 1, 100))
    interpolated_after = f_after(np.linspace(0, len(after_middle) - 1, 100))

    # Combine the interpolated part with the original time series
    interpolated_time_series = np.zeros((201, ))
    
    interpolated_time_series[:100] = interpolated_before
    interpolated_time_series[100] = time_series[middle_index]
    interpolated_time_series[101:] = interpolated_after

    return interpolated_time_series
# =============================================================================   


def seasonal_autocorrelation(data, seasonal_period):
    autocorrelations = []
    for i in range(seasonal_period):
        seasonal_data = data[i::seasonal_period]
        mean = np.mean(seasonal_data)
        variance = np.var(seasonal_data, ddof=1)
        covariance = np.sum((seasonal_data[:-1] - mean) * (seasonal_data[1:] - mean))
        if variance < np.finfo(np.float32).eps:
            autocorrelation = covariance / (variance + np.finfo(np.float32).eps)
        else:    
            autocorrelation = covariance / variance
        autocorrelations.append(autocorrelation)

    autocorrelation_coefficient = np.nanmean(autocorrelations)

    return autocorrelation_coefficient
# =============================================================================   


def spectral_edge_frequency(power, freqs, edges=[0.5, 0.7, 0.8, 0.9, 0.95]):
    out = np.cumsum((power))
    out = out / out[-1]
    ret = []
    if np.sum(np.isnan(out))>0:
        ret = np.zeros((len(edges), ))
    else:
        for edge in edges:
            ret.append(freqs[np.where(out>edge)[0][0]])
        ret = np.array(ret)
    return ret
# =============================================================================   


def bandpass_filter(data, sampling_frequency, passband=[0.1, 45.0]):
    """
    data: rows are ECG waveforms, columns are samples
    sampling_frequency: in Hz
    passband: pass-band frequency range
    """
    data = np.asarray(data, dtype=np.float64)
    data = mne.filter.filter_data(data, sampling_frequency, passband[0],
                                  passband[1], n_jobs=4, verbose='error')
    return data
# =============================================================================   


def zscore(data):
    """
    data: rows are ECG waveforms, columns are samples
    """
    row_means = np.nanmean(data, axis=1, keepdims=True)
    row_stds = np.nanstd(data, axis=1, keepdims=True)
    data = (data - row_means) / row_stds
    return data
# =============================================================================   


def plot_waveform(waveform, sampling_frequency, title="ECG"):
    # Calculate the time vector
    num_samples = len(waveform)
    time_vector = np.arange(num_samples) / sampling_frequency
    
    plt.figure()
    plt.plot(time_vector, waveform, label=title)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
# =============================================================================   

    
def plot_random_waveforms(waveforms, sampling_frequency, num_samples=9, title="ECG", grid=True):
    # Calculate the time vector
    num_total_samples = waveforms.shape[1]
    time_vector = np.arange(num_total_samples) / sampling_frequency
    
    # Randomly select samples
    sample_indices = np.random.choice(num_total_samples, size=num_samples, replace=False)
    samples = waveforms[sample_indices, :]
    
    # Plotting
    num_rows = num_samples // 3 + (1 if num_samples % 3 != 0 else 0)
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    fig.suptitle(title)
    
    for i in range(samples.shape[0]):
        sample = samples[i, :]
        row = i // 3
        col = i % 3
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        ax.plot(time_vector, sample)
        ax.set_title(f'Sample {sample_indices[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(grid)
    
    plt.tight_layout()
    plt.show()
# =============================================================================   


def melspectrogram(x, sampling_freq, display=False, title='ECG'):
    
    # RAW SPECTROGRAM
    mel_spec = librosa.feature.melspectrogram(y=x, sr=sampling_freq, hop_length=len(x)//256, 
          n_fft=1024, n_mels=128, fmin=0, fmax=40, win_length=128)

    # LOG TRANSFORM
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

    # STANDARDIZE TO -1 TO 1
    mel_spec_db = (mel_spec_db+40)/40
    
    if display:
        plt.figure()
        plt.imshow(mel_spec_db,aspect='auto',origin='lower')
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Magnitude (standardized dB)')

    return mel_spec_db
# =============================================================================   


def next_power_of_2(x):
    return int(2 ** np.ceil(np.log2(x)))
# ============================================================================= 

  
def spectrogram(x, sampling_freq, window_length=[], overlap=[], display=False):
    
    if isinstance(window_length, list):
        window_length = int(sampling_freq*3)
    if isinstance(overlap, list):
        overlap = int(sampling_freq*2)
        
    n_samples = len(x)
    window = np.hamming(window_length)
    hop_size = window_length - overlap
    n_fft = next_power_of_2(window_length)

    n_windows = int(np.floor((n_samples - window_length) / hop_size) + 1)
    spectrogram = np.zeros((n_fft // 2 + 1, n_windows))

    for i in range(n_windows):
        start = i * hop_size
        end = start + window_length
        segment = x[start:end]

        windowed_segment = segment * window
        spectrum = np.fft.fft(windowed_segment, n=n_fft)
        magnitude = np.abs(spectrum[:n_fft // 2 + 1])
        spectrogram[:, i] = magnitude

    frequencies = np.fft.fftfreq(n_fft, d=1/sampling_freq)[:n_fft // 2 + 1]
    time = np.arange(n_windows) * (hop_size / sampling_freq)

    if display:
        plt.figure()
        plt.imshow(10 * np.log10(spectrogram), aspect='auto', origin='lower',
                   extent=[time.min(), time.max(), frequencies.min(), frequencies.max()])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        plt.colorbar(label='Magnitude (dB)')
        plt.show()

    return spectrogram, frequencies, time
# =============================================================================   


def plot_random_waveforms_spectrogram(waveforms, sampling_frequency, num_samples=9, title="ECG", grid=True):
    # Calculate the time vector
    num_total_samples = waveforms.shape[1]
    time_vector = np.arange(num_total_samples) / sampling_frequency
    
    # Randomly select samples
    sample_indices = np.random.choice(num_total_samples, size=num_samples, replace=False)
    samples = waveforms[sample_indices, :]
    
    # Plotting
    num_rows = num_samples // 3 + (1 if num_samples % 3 != 0 else 0)
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    fig.suptitle(title)
    
    for i in range(samples.shape[0]):
        sample = samples[i, :]
        row = i // 3
        col = i % 3
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        ax.plot(time_vector, sample)
        ax.set_title(f'Sample {sample_indices[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(grid)
        
        # Plot Spectrogram
        spectrogram1, frequencies, time = spectrogram(sample, sampling_frequency)
        ax_spec = ax.twinx()
        ax_spec.imshow(10 * np.log10(spectrogram1), aspect='auto', origin='lower',
                       extent=[time.min(), time.max(), frequencies.min(), frequencies.max()])
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_ylim(frequencies.min(), frequencies.max())
    
    plt.tight_layout()
    plt.show()