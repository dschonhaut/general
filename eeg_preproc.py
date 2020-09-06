"""
manning_analysis.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com

Description: 
    Functions that perform the core data processing and analyses for the Manning
    et al., J Neurosci 2009 paper.

Last Edited: 
    6/21/19
"""
import os

import numpy as np
from scipy.interpolate import interp1d

import mne
from ptsa.data.TimeSeriesX import TimeSeriesX as TimeSeries
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
    
def process_lfp(lfp, # chan x time
                subj_sess, 
                chans, # str channels starting at '1'
                sampling_rate, # sampling rate in Hz
                resampling_rate=0,
                notch_freqs=None, # list of freqs (in Hz) to notch filter
                l_freq=None,
                h_freq=None,
                mean_sub=False,
                interpolate=False, 
                session_spikes=None, 
                ms_before=2, 
                ms_after=4):
    """Notch filter the raw LFP data and linearly interpolate around spikes.
    
    The raw data are returned in timeseries form by default.
    
    Parameters
    ----------
    lfp : numpy.ndarray or ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints array with the raw LFP data.
        
    Returns
    -------
    lfp : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints array with the processed LFP data. 
    """
    
    if sampling_rate == resampling_rate:
        resampling_rate = 0
    if resampling_rate > sampling_rate:
        print('CANNOT UPSAMPLE')
        assert False
    if interpolate and resampling_rate > 0:
        print('FIX SPIKE INTERPOLATION OF RESAMPLED DATA!')
        assert False
        
    if type(lfp) == np.ndarray:
        lfp = TimeSeries(lfp, name=subj_sess, 
                         dims=['channel', 'time'],
                         coords={'channel': chans,
                                 'time': np.arange(lfp.shape[1]),
                                 'samplerate': sampling_rate})                                 
    process_tag = ''
    
    # Resample.
    if resampling_rate > 0:
        lfp = resample(lfp, resampling_rate)
        old_sampling_rate = sampling_rate
        sampling_rate = resampling_rate
        process_tag += 'resample-{}Hz_'.format(int(sampling_rate))
    
    # Notch filter using an FIR filter.
    if notch_freqs:
        lfp.data = mne.filter.notch_filter(np.float64(lfp.copy().data), 
                                           Fs=sampling_rate, 
                                           freqs=notch_freqs,
                                           #notch_widths=2,
                                           phase='zero',
                                           verbose=False)
        process_tag += 'notch-filt_'
    
    # Bandpass filter the data.
    if l_freq or h_freq:
        lfp = mne.filter.filter_data(np.float64(lfp.copy().data), 
                                          sfreq=sampling_rate, 
                                          l_freq=l_freq, 
                                          h_freq=h_freq,
                                          verbose=False)
        if l_freq:
            process_tag += 'lfreq{}Hz_'.format(l_freq)
        if h_freq:
            process_tag += 'hfreq{}Hz_'.format(h_freq)
    
    # Mean subtract each channel over time.
    if mean_sub:
        lfp.data = lfp.data - np.expand_dims(np.mean(lfp.data, axis=1), axis=-1)
        process_tag += 'meansub_'
    
    # Linearly interpolate LFP around spikes, for channels with clusters.
    if interpolate:
        interp_chans = session_spikes.keys()
        for chan in interp_chans:
            interp_mask = session_spikes[chan]['interp_mask']
            keep_inds = np.where(interp_mask==0)[0]
            fill_inds = np.where(interp_mask==1)[0]
            f = interp1d(keep_inds, lfp.data[chans.index(chan), keep_inds],
                         kind='linear', fill_value='extrapolate')
            def apply_interp(arr, inds, f):
                arr[inds] = f(inds)
                return arr
            lfp.data[chans.index(chan), :] = apply_interp(
                    lfp.data[chans.index(chan), :], 
                    fill_inds, 
                    f
                )
        process_tag += 'spikeinterpolation-{}to{}ms_'.format(ms_before, ms_after)
         
    if process_tag:
        process_tag = process_tag[:-1]
    else:
        process_tag = 'same as lfp_raw'
        
    lfp.attrs={'process_tag': process_tag}
        
    return lfp

def resample(lfp, new_sampling_rate, events=None):
    """Resample LFP data using MNE.
    
    Parameters
    ----------
    lfp : ptsa.data.timeseries.TimeSeries
        TimeSeries object containing an LFP data array.
    new_sampling_rate : int or float
        The sampling rate that the data is resampled to.
    events : np.ndarray
        1-d or 2-d array of event time indices at the
        old sampling rate.
    
    Returns
    -------
    lfp : ptsa.data.timeseries.TimeSeries
        TimeSeries object containing the resampled LFP data 
        array.
    events : np.ndarray
        1-d or 2-d array of event time indices at the
        new sampling rate.   
    """
    chans = lfp.channel.values.tolist()
    old_sampling_rate = lfp.samplerate.data.tolist()
    info = mne.create_info(ch_names=chans, 
                           sfreq=old_sampling_rate, 
                           ch_types=['seeg']*len(lfp.channel.data))
    lfp_mne = mne.io.RawArray(lfp.copy().data, info, verbose=False)
    lfp_mne = lfp_mne.resample(new_sampling_rate)
    
    lfp = TimeSeries(lfp_mne.get_data(), 
                     name=lfp.name,
                     dims=['channel', 'time'],
                     coords={'channel': chans,
                             'time': np.arange(lfp_mne.get_data().shape[1]),
                             'samplerate': new_sampling_rate})
    
    if events is not None:
        ratio = new_sampling_rate / old_sampling_rate
        events = np.round(events * ratio).astype(int)
        return lfp, events
    else:
        return lfp
        
def run_morlet(timeseries, 
               freqs=None, 
               width=5, 
               output=['power', 'phase'],
               log_power=False, 
               z_power=False, 
               z_power_acrossfreq=False, 
               overwrite=False,
               savedir='/data3/scratch/dscho/frLfp/data/lfp/morlet',
               power_file=None,
               phase_file=None,
               verbose=False):
    """Apply Morlet wavelet transform to a timeseries to calculate
    power and phase spectra for one or more frequencies.
    
    Serves as a wrapper for PTSA's MorletWaveletFilter. Can log 
    transform and/or Z-score power across time and can save the 
    returned power and phase timeseries objects as hdf5 files.
    
    Parameters
    ----------
    timeseries : ptsa.data.timeseries.TimeSeries
        The timeseries data to be transformed.
    freqs : numpy.ndarray or list
        A list of frequencies to apply wavelet decomposition over.
    width : int
        Number of waves for each frequency.
    output : str or list or numpy.ndarray
        ['power', 'phase'], ['power'], or ['phase'] depending on
        what output is desired.
    log_power : bool
        If True, power values are log10 transformed.
    z_power : bool
        If True, power values are Z-scored across the time dimension.
        Requires timeseries to have a dimension called 'time'.
        z_power and z_power_acrossfreq can't both be True.
    z_power_acrossfreq : bool
        If True, power values are Z-scored across frequencies and
        time for a given channel. Requires timeseries to have
        a dimension called 'time'. z_power and z_power_acrossfreq 
        can't both be True.
    overwrite : bool
        If True, existing files will be overwritten.
    savedir : str
        Directory where the output files (power and phase timeseries
        objects saved in hdf5 format) will be saved. No files are
        saved if savedir is None.
    verbose : bool
        If verbose is False, print statements are suppressed.

    Returns
    -------
    power : ptsa.data.timeseries.TimeSeries
        Power spectra with optional log and/or Z transforms applied.
        Has the same shape as timeseries. 
    phase : ptsa.data.timeseries.TimeSeries
        Phase spectra with optional log and/or Z transforms applied.
        Has the same shape as timeseries.
    """
    dims = timeseries.dims
    dim1 = dims[0]
    assert len(dims) == 2
    assert dims[1] == 'time'
    assert not np.all([z_power, z_power_acrossfreq])
    
    if type(output) == str:
        output = [output]
    assert 'power' in output or 'phase' in output
    
    if freqs is None:
        freqs = np.logspace(np.log10(2), np.log10(200), 50, base=10)
        
    fstr = ('_width{}_{:.0f}-{:.0f}Hz-{}log10steps'
            .format(width, min(freqs), max(freqs), len(freqs)))
            
    powfstr = ''
    if log_power:
        powfstr += '_log10'
    if z_power:
        powfstr += '_Z-withinfreq'
    if z_power_acrossfreq:
        powfstr += '_Z-acrossfreq'
    
    # If power and phase already exist and aren't supposed to be overwritten,
    # load and return them from disk space.
    if savedir:
        if power_file is None:
            fname = ('{}_ch{}_power{}{}.hdf'
                     .format(timeseries.name, timeseries[dim1].data[0], fstr, powfstr))
            power_file = os.path.join(savedir, fname)
        if phase_file is None:
            fname = ('{}_ch{}_phase{}.hdf'
                     .format(timeseries.name,timeseries[dim1].data[0], fstr))
            phase_file = os.path.join(savedir, fname)
            
        if len(output) == 2:
            files_exist = os.path.exists(power_file) and os.path.exists(phase_file)
        elif 'power' in output:
            files_exist = os.path.exists(power_file)
        else:
            files_exist = os.path.exists(phase_file)
            
        if files_exist and not overwrite:
            if len(output) == 2:
                if verbose:
                    print('Loading power and phase:\n\t{}\n\t{}'
                          .format(power_file, phase_file))
                power = TimeSeries.from_hdf(power_file)
                phase = TimeSeries.from_hdf(phase_file)
                return power, phase
            elif 'power' in output:
                if verbose:
                    print('Loading power:\n\t{}'
                          .format(power_file))
                power = TimeSeries.from_hdf(power_file)
                return power
            else:
                if verbose:
                    print('Loading phase:\n\t{}'
                          .format(phase_file))
                phase = TimeSeries.from_hdf(phase_file)
                return phase
    
    # Get power and phase.
    if len(output) == 2:
        if verbose:
            print('Calculating power and phase.')
        power, phase = MorletWaveletFilter(timeseries,
                                           freqs=freqs,
                                           width=width,
                                           output=['power', 'phase']).filter()
    elif 'power' in output:
        if verbose:
            print('Calculating power.')
        power = MorletWaveletFilter(timeseries,
                                    freqs=freqs,
                                    width=width,
                                    output=['power']).filter()
    else:
        if verbose:
            print('Calculating phase.')
        phase = MorletWaveletFilter(timeseries,
                                    freqs=freqs,
                                    width=width,
                                    output=['phase']).filter()                  
        
        
    if 'power' in output:
        power = TimeSeries(power.data, dims=['frequency', dim1, 'time'], 
                           name=timeseries.name, 
                           coords={'frequency': power.frequency.data,
                                   dim1: power[dim1].data,
                                   'time': power.time.data,
                                   'samplerate': power.samplerate.data},
                           attrs={'morlet_width': width})
                           
         # Log transform every power value.
        if log_power:
            if verbose: 
                print('Log-transforming power values.')
            power.data = np.log10(power)
            
        # Z-score power over time for each channel, frequency vector
        if z_power:
            if verbose:
                print('Z-scoring power across time, within each frequency.')
            power.data = (power - power.mean(dim='time')) / power.std(dim='time')
        
        # Z-score power across frequencies and time, for each channel
        if z_power_acrossfreq:
            if verbose:
                print('Z-scoring power across time and frequency.')
            power.data = ((power - power.mean(dim=['frequency', 'time'])) 
                          / power.std(dim=['frequency', 'time']))
    
    if 'phase' in output:                         
        phase = TimeSeries(phase.data, dims=['frequency', dim1, 'time'], 
                           name=timeseries.name, 
                           coords={'frequency': phase.frequency.data,
                                   dim1: phase[dim1].data,
                                   'time': phase.time.data,
                                   'samplerate': phase.samplerate.data},
                           attrs={'morlet_width': width})     
    
    # Return log-transformed power and phase.
    if savedir:
        if verbose:
            print('Saving power:\n\t{}'.format(power_file))
        power.to_hdf(power_file)
    
    if len(output) == 2:
        return power, phase
    elif 'power' in output:
        return power
    else:
        return phase