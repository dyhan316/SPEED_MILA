import mne
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import logging
import h5py
from collections import OrderedDict
from datetime import datetime
from time import sleep
from src.utils import split_raw, make_tuh_montage
from src.methods import PreprocessMethods

# Import typing
from typing import Tuple, List, Optional, Dict, Union
import sys

# Make ABC abstract class
from abc import ABC, abstractmethod

class Pipeline(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
    
class BasePipeline(Pipeline):
    """
    A class for preprocessing EEG data files.

    Parameters:
    - src_paths (List[str]): Paths to the source EEG files.
    - window_length (int): Length of the window for processing, in seconds. Default is 60.
    - hp_freq (float): High pass filter frequency. Default is 1.0.
    - lp_freq (float): Low pass filter frequency. Default is 100.0.
    - line_freqs (List[float]): Frequencies to be removed by the notch filter. Default is [60.0].
    - iclabel_threshold (float): Threshold for IC label classification. Default is 0.7.
    - min_nchans (int): Minimum number of channels. Default is 10.
    - do_ica (bool): Whether to perform ICA. Default is True.
    - included_components (List[str]): Components to include. Default is ["brain", "other"].
    - memory_efficient (bool): If True, uses a memory-efficient approach. Default is True.
    """
    def __init__(
            self, 
            window_length: int = 60,
            sfreq: float = 256.0,
            hp_freq: Optional[float] = 0.5, 
            lp_freq: Optional[float] = 100.0,
            line_freqs: List[float] = [60.0], 
            iclabel_threshold: float = 0.7,
            quality_check: bool = True,
            min_nchans: int = 10, 
            do_ica: bool = True, 
            included_components: List[str] = ["brain", "other"], 
            memory_efficient: bool = True,
            montage_name: str = "tuh",
            channels: List[str] = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'], #! CHANGE CHANNELS HERE!!
            channels_rename: Optional[Dict[str, str]] = None,
            #! ADDED PARAMETERS FOR US 
            keep_remainder : bool = False, #if True, will keep the remainder of the data after splitting (last window will be longer due to the added thing)
            run_ransac : bool = True, #if False, will not do ransac during QA
            run_drop_bad_channels : bool = True,
            run_detrend : bool = True, #if False, will not detrend during preprocessing (meegkit detrending)
            preproc_lp_freq : Union[float, str] = "same", #(during preproc) if None : no high pass, if 'same', will use the same lp_freq for QA and preprocessing.  
            preproc_hp_freq : Union[float, str] = "same", #(during preproc) if None, no low pass, if 'same', will use the same hp_freq for QA and preprocessing
            preproc_line_noise_option : Union[float, str] = "default" , #if default, will use the default line noise removal option, if mnenotch, use MNE's default, if False/None, will not remove line noise
            preproc_line_freqs : Union[List[float], None, str] = "same", #(during preproc) if same, will use the same line_freqs for QA and preprocessing
            rereference_option : Union[str, bool] = 'ar', #what rereference option to use for preprocessing. ar, le, or (None or False) (no rereferencing)
            missing_chs_option : str = 'interpolate', #what to do with missing channels. 1. interpolate (default), 2. zero, 3. remove (remove window), or 4. ignore (keep on despite the missing channels)
            drop_extra_chs : bool = True, #if False, will not drop extra channels
            reorder_chs : bool = True, #if False, will not reorder channels
            resample_method : Union[str, bool] = 'default', #options : 'default' (SPEED's interpolate_nearest), 'mne' (MNE's interpolate), 'decimate' (not implemented yet), False/None (no resampling)
        ):
        
        mne.set_log_level('ERROR')
        
        self.channels_rename = channels_rename
        
        self.sfreq = sfreq
        self.interpolation_mode = "accurate"
        self.quality_check = quality_check
        
        # Quality check thresholds
        self.oha_threshold = 40e-6
        self.thv_threshold = 40e-6
        self.chv_threshold = 80e-6
        self.min_unique = 100
                            
        self.memory_efficient = memory_efficient        
        self._setup_montage_and_channels(montage_name, channels)
        self._setup_filters(hp_freq, lp_freq, line_freqs)
        self._setup_ica(iclabel_threshold, included_components, do_ica)

        self.min_nchans = min_nchans
        self.window_length = window_length # Seconds 
        
        #! =================ADDED PARAMETERS FOR US=================
        ### 1. splitting windows and QA options 
        self.keep_remainder = keep_remainder
        self.ransac = run_ransac 
        self.run_drop_bad_channels = run_drop_bad_channels
        ### 2. Actual Preprocessing if 'same', will use same lp,hp,line_freq for QA and preprocessing
        self.run_detrend = run_detrend
        
        # Preprocessing line noise option
        if preproc_line_noise_option in ['default', 'mnenotch']:
            self.preproc_line_noise_option = preproc_line_noise_option
        elif preproc_line_noise_option in [None, 'None', False]:
            self.preproc_line_noise_option = None
        else:
            raise ValueError("Invalid preproc_line_noise_option. Allowed values are 'default', 'mnenotch', None, or False.")

        # Preprocessing line frequencies
        if preproc_line_freqs == 'same':
            self.preproc_line_freqs = line_freqs
        elif preproc_line_freqs in [None, 'None', False]:
            self.preproc_line_freqs = []
        elif isinstance(preproc_line_freqs, list):
            self.preproc_line_freqs = preproc_line_freqs
        else:
            raise ValueError("Invalid preproc_line_freqs. Provide a list of frequencies, 'same', or set to None/False.")

        # Preprocessing low-pass frequency
        if preproc_lp_freq == 'same':
            self.preproc_lp_freq = lp_freq
        elif preproc_lp_freq in [None, 'None', False]:
            self.preproc_lp_freq = None
        elif isinstance(preproc_lp_freq, (int, float)):
            self.preproc_lp_freq = preproc_lp_freq
        else:
            raise ValueError("Invalid preproc_lp_freq. Provide a frequency value, 'same', or set to None/False.")

        # Preprocessing high-pass frequency
        if preproc_hp_freq == 'same':
            self.preproc_hp_freq = hp_freq
        elif preproc_hp_freq in [None, 'None', False]:
            self.preproc_hp_freq = None
        elif isinstance(preproc_hp_freq, (int, float)):
            self.preproc_hp_freq = preproc_hp_freq
        else:
            raise ValueError("Invalid preproc_hp_freq. Provide a frequency value, 'same', or set to None/False.")

        ### 3. rereference,  interpolating missing channels, dropping extra channels, and resampling 
        # Rereference option
        if rereference_option in ['ar', 'le']:
            self.rereference_option = rereference_option
        elif rereference_option in [None, 'None', False]:
            self.rereference_option = None
        else:
            raise ValueError("Invalid rereference_option. Allowed values are 'ar', 'le', None, or False.")
        self.missing_chs_option = missing_chs_option
        self.drop_extra_chs = drop_extra_chs
        self.reorder_chs = reorder_chs
        if resample_method in ['default', 'mne']:
            self.resample_method = resample_method
        elif resample_method == ["decimate"] : 
            raise NotImplementedError("Decimate method is not implemented yet.")
        elif resample_method in ["None", None, False]:
            self.resample_method = None
        else :
            raise ValueError("Invalid resample_method. Allowed values are 'default', 'mne', or 'decimate'.")
        
        #sanity check
        if reorder_chs : 
            assert missing_chs_option in ['interpolate', 'zero', 'remove'], "If reorder_chs is True, missing_chs_option must be one of ['interpolate', 'zero', 'remove'] so that the channel # is consistent!"
            assert drop_extra_chs, "If reorder_chs is True, drop_extra_chs must be True as well."

    def _setup_montage_and_channels(self, montage_name, chs):
        """Setup montage and channels."""
        self.chs = chs
        self.montage_name = montage_name
        
        if self.montage_name == "tuh":
            montage = make_tuh_montage()            
        else:
            montage = mne.channels.make_standard_montage(self.montage_name)
                    
        self.montage = montage
        
    def _setup_filters(self, hp_freq, lp_freq, line_freqs):
        """Configure filters and window length."""
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq
        self.line_freqs = np.sort([line_freqs]).flatten()
        
    def _setup_ica(self, iclabel_threshold, included_components, do_ica):
        """
        Setup Independent Component Analysis (ICA) to identify and exclude artifacts.
        """
        self._possible_iclabels = ["brain", "muscle", "eye", "heart", "line noise", "channel noise", "other"]
        self.included_components = included_components
        assert all([label in self._possible_iclabels for label in self.included_components])
        
        self.iclabel_threshold = iclabel_threshold
        self.do_ica = do_ica
        
    # Add typing
    def _split_raw(self, raw: mne.io.Raw):
        return split_raw(raw, self.window_length, self.keep_remainder)
    
    def _set_montage(self, raw: mne.io.Raw):
        return PreprocessMethods.set_montage(raw, self.montage)
    
    def _to_standard_names(self, raw: mne.io.Raw):
        return PreprocessMethods.to_standard_names(raw)
    
    def _average_reference(self, raw: mne.io.Raw):
        return mne.set_eeg_reference(raw, ref_channels='average', projection=False, copy=False, verbose=False)
        
    def _linkedear_reference(self, raw: mne.io.Raw):
        return mne.set_eeg_reference(raw, ref_channels=['A1','A2'], projection=False, copy=False, verbose=False)
    
    def _evaluate_quality(self, raw: mne.io.Raw):
        if not self.quality_check:
            return True
        else:
            return PreprocessMethods.evaluate_quality(raw, self.oha_threshold, self.thv_threshold, self.chv_threshold, self.min_unique, 
                              self.min_nchans, self.line_freqs, self.hp_freq, self.lp_freq)
                    
    def _interpolate_nearest(self, raw: mne.io.Raw):     
        return PreprocessMethods.interpolate_nearest(raw, self.sfreq)

    def _do_resampling(self, raw: mne.io.Raw, resample_method : Optional[str] = 'default'):
        if resample_method == 'default':
            return self._interpolate_nearest(raw)
        elif resample_method == 'mne':
            return raw.resample(self.sfreq, npad="auto", verbose=False)
        elif resample_method == 'decimate':
            raise NotImplementedError("Decimate method is not implemented yet.")
        elif resample_method in ["None", None, False]:
            return raw
        else:
            raise ValueError("Invalid resample_method. Allowed values are 'default', 'mne', or 'decimate'.")            
        
    def _drop_bad_channels(self, raw: mne.io.Raw):       
        return PreprocessMethods.find_bad_channels(raw, ransac = self.ransac, drop = True)

    def _filter(self, raw: mne.io.Raw):
        return PreprocessMethods.filter(raw, self.preproc_hp_freq, self.preproc_lp_freq, [], do_detrend=self.run_detrend, preproc_line_noise_option = False)
    
    def _remove_line_noise(self, raw: mne.io.Raw, preproc_line_noise_option = "default"):
        return PreprocessMethods.filter(raw, None, None, self.preproc_line_freqs, do_detrend=False, preproc_line_noise_option = preproc_line_noise_option)

    def _ica_clean(self, raw: mne.io.Raw):
        return PreprocessMethods.ica_clean(raw, self.iclabel_threshold, self.included_components)       

    def _interpolate_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.interpolate_missing(raw, self.chs, self.montage, mode=self.interpolation_mode)
    
    def _zero_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.zero_missing(raw, self.chs, self.montage)
    
    def _handle_missing_chs(self, raw: mne.io.Raw, missing_chs_option: str):
        if missing_chs_option == 'interpolate':
            return self._interpolate_missing(raw)
        elif missing_chs_option == 'zero':
            return self._zero_missing(raw)
        elif missing_chs_option == 'remove':
            missing_ch = [c for c in self.chs if c not in raw.ch_names]
            #don't do anything if there are no missing channels
            if len(missing_ch) == 0: 
                return missing_ch
            else :  
                raw = None
                logging.error(f"Dropping window due to missing channels {missing_ch}.")
                raise ValueError(f"Missing channels: {missing_ch}.")
        elif missing_chs_option == 'ignore':
            return [c for c in self.chs if c not in raw.ch_names]        
        else : 
            sys.exit("CRITICAL ERROR: Invalid missing channel option. Must be one of ['interpolate', 'zero', 'remove', 'ignore']")  
            #not use ValueError becuae this is a critical error that should kill everything (for loop of try statements should be exited all together)
    #!below : not used anymore, just use _drop_extra and _reorder_chs separately 
    def _drop_extra_and_reorder(self, raw: mne.io.Raw):
        return PreprocessMethods.drop_extra_and_reorder(raw, self.chs)
    
    def _drop_extra_chs(self, raw: mne.io.Raw):
        return PreprocessMethods.drop_extra_chs(raw, self.chs)

    def _reorder_chs(self, raw: mne.io.Raw):
        return PreprocessMethods.reorder_chs(raw, self.chs)

class DynamicPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        return self.run(src_paths)    

    def run(self, src_paths):
        logging.debug("Loading EDF files...")
        src_paths = [Path(src_path) for src_path in src_paths]
        raws = []
        times = []
        indices = []
        
        logging.debug("Splitting raws...")   
        for i, src_path in enumerate(src_paths):
            try:
                raw_orig = mne.io.read_raw_edf(src_path, preload=True, verbose=False)
                
                # Rename channels with channel_rename
                if self.channels_rename is not None:
                    raw_orig.rename_channels(self.channels_rename)
                    logging.info(f"File: {src_paths[i].stem}.\tRenamed channels: {self.channels_rename}.")
                
                # Skip if raw_orig is too short
                if raw_orig.times[-1] < 1.5*self.window_length:
                    logging.info(f"File: {src_path.stem}.\tDuration too short. Skipping file.")
                    continue
                
                self._to_standard_names(raw_orig) # "EEG FP1-LE" -> "Fp1"... 
                drop_chs = self._set_montage(raw_orig) #! STILL KEEPS ALL EEG (even if not in the 19) drops channels from raw_orig and returns what was dropped
                logging.info(f"File: {src_paths[i].stem}.\tDropped {len(drop_chs)} channels when setting montage: {drop_chs}.")
            except Exception as e:
                print(f"====ERROR File: {src_path.stem}.\tError: {e}====")
                logging.error(f"Dropping file: {src_path.stem}.\tError: {e}")
                continue
            # AT THIS POINT : NOTHING MORE THAN DROPPOING NON BRAIN CHANNELS ARE DONE (NO PREPROC YET)
            raws_split, times_split = self._split_raw(raw_orig)
            raws.extend(raws_split)
            times.extend(times_split)
            indices.extend([i] * len(raws_split)) #indices : doesn't matter if batchsize = 1.. used for keeping track of what same edf files are 
            
        total_windows = len(raws)
        logging.debug(f"Total windows: {total_windows}")
        
        for i, raw in enumerate(raws):
            start_time = round(times[i][0], 1)
            end_time = round(times[i][1], 1)
            filename = src_paths[indices[i]].stem            
            
            try:
                raws[i] = self.run_single(raw, start_time, end_time, filename)
            except Exception as e:
                logging.error(f"File: {filename}.\tTime: {(start_time, end_time)}.\tError: {e}")
                raws[i] = None
                
            # Log progress every N windows
            if (i + 1) % 10 == 0:
                logging.debug(f"Processed {i + 1}/{total_windows} windows.")
        return raws, times, indices
        
    def run_single(self, raw, start_time, end_time, filename) -> Optional[mne.io.Raw]:
        window_info_str = f"File: {filename}.\tTime: {(start_time, end_time)}."
               
        #! BELOW : QUALITY 는 "window" removal! (channel싹다 제거)
        quality = self._evaluate_quality(raw)
        if not quality:
            raw = None
            logging.info(f"{window_info_str}\tQuality check failed. Dropping window.")
            return None
        
        self._remove_line_noise(raw, preproc_line_noise_option = self.preproc_line_noise_option)
        if self.run_drop_bad_channels:
            bad_chs = self._drop_bad_channels(raw)
            logging.info(f"{window_info_str}\tFound {len(bad_chs)} bad channels: {bad_chs}.")

        quality = self._evaluate_quality(raw)
        if not quality:
            raw = None
            logging.info(f"{window_info_str}\tQuality check failed. Dropping window.")
            return None
        
        self._filter(raw)    
        if self.rereference_option is not None:
            if self.rereference_option == 'ar':
                self._average_reference(raw)
            elif self.rereference_option == 'le':
                self._linkedear_reference(raw)
            else:
                raise ValueError(f"Invalid rereference option. Please provide one of ['ar', 'le'] or None/False.")
                
        if self.do_ica:
            print("ICA")
            excluded_idxs, labels, y_proba = self._ica_clean(raw)
            logging.info(f"{window_info_str}\tExcluding {len(excluded_idxs)} components: {excluded_idxs}.")
            logging.info(f"{window_info_str}\tLabels: {labels}.")
            logging.info(f"{window_info_str}\tProbabilities: {[round(prob, 2) for prob in y_proba]}.")
            
            bad_chs = self._drop_bad_channels(raw)
            logging.info(f"{window_info_str}\tFound {len(bad_chs)} bad channels: {bad_chs}.")

        missing_chs = self._handle_missing_chs(raw, self.missing_chs_option)
        logging.info(f"{window_info_str}\t Missing channel - {self.missing_chs_option}-ing {len(missing_chs)} channels: {missing_chs}.")            
        if self.drop_extra_chs:
            extra_chs = self._drop_extra_chs(raw)
            logging.info(f"{window_info_str}\tDropping {len(extra_chs)} extra channels: {extra_chs}.")
        if self.reorder_chs:
            _ = self._reorder_chs(raw)
            logging.info(f"{window_info_str}\tReordering channels.")
        
        self._do_resampling(raw, self.resample_method) # self._interpolate_nearest(raw)
        
        return raw
    
class DynamicPipelineBENDR(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        return self.run(src_paths)   
    
    def _zero_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.zero_missing(raw, self.chs, self.montage)

    def run(self, src_paths):
        logging.debug("Loading EDF files...")
        src_paths = [Path(src_path) for src_path in src_paths]

        raws = []
        times = []
        indices = []
        
        logging.debug("Splitting raws...")   
        for i, src_path in enumerate(src_paths):
            try:
                raw_orig = mne.io.read_raw_edf(src_path, preload=True, verbose=False)
                
                # Skip if raw_orig is too short
                if raw_orig.times[-1] < 1.5*self.window_length:
                    logging.info(f"File: {src_path.stem}.\tDuration too short. Skipping file.")
                    continue
                
                self._to_standard_names(raw_orig)
                drop_chs = self._set_montage(raw_orig)
                logging.info(f"File: {src_paths[i].stem}.\tDropped {len(drop_chs)} channels when setting montage: {drop_chs}.")
            except Exception as e:
                logging.error(f"Dropping file: {src_path.stem}.\tError: {e}")
                continue
                
            raws_split, times_split = self._split_raw(raw_orig)
            raws.extend(raws_split)
            times.extend(times_split)
            indices.extend([i] * len(raws_split))
            
        total_windows = len(raws)
        logging.debug(f"Total windows: {total_windows}")
                       
        for i, raw in enumerate(raws):
            start_time = round(times[i][0], 1)
            end_time = round(times[i][1], 1)
            filename = src_paths[indices[i]].stem            
            
            try:
                raws[i] = self.run_single(raw, start_time, end_time, filename)
            except Exception as e:
                logging.error(f"File: {filename}.\tTime: {(start_time, end_time)}.\tError: {e}")
                raws[i] = None
                
            # Log progress every N windows
            if (i + 1) % 10 == 0:
                logging.debug(f"Processed {i + 1}/{total_windows} windows.")
        
        mask = [True if raw is not None else False for raw in raws]
        raws = [raw for raw, m in zip(raws, mask) if m]
        times = [time for time, m in zip(times, mask) if m]
        indices = [idx for idx, m in zip(indices, mask) if m]
        
        return raws, times, indices
        
    def run_single(self, raw, start_time, end_time, filename) -> Optional[mne.io.Raw]:
        window_info_str = f"File: {filename}.\tTime: {(start_time, end_time)}."       

        missing_chs = self._zero_missing(raw)
        logging.info(f"{window_info_str}\Zeroing {len(missing_chs)} channels: {missing_chs}.")
        
        extra_chs = self._drop_extra_and_reorder(raw)
        logging.info(f"{window_info_str}\tRemoving {len(extra_chs)} extra channels: {extra_chs}.")
        
        self._interpolate_nearest(raw)
        
        return raw

# ToDo: Remove dropped windows from objects        
class StaticPipeline(BasePipeline):
    def __init__(self, src_paths: List[str], **kwargs):
        super().__init__(**kwargs)
        self.src_paths = [Path(src_path) for src_path in src_paths]
        self.raws_orig = [mne.io.read_raw_edf(src_path, preload=True, verbose=False) for src_path in self.src_paths]
        
        self.dropped_chs = [None] * len(self.src_paths)
        
        self.raws = []
        self.times = []
        self.indices = []        
                
        for i, raw_orig in enumerate(self.raws_orig):
            self._to_standard_names(raw_orig)
            self.dropped_chs[i] = self._set_montage(raw_orig)
            raws_split, times_split = self._split_raw(raw_orig)
            self.raws.extend(raws_split)
            self.times.extend(times_split)
            self.indices.extend([i] * len(raws_split))
    
        self.bad_chs =  [[] for _ in range(len(self.raws))]
        self.missing_chs = [[] for _ in range(len(self.raws))]
        self.extra_chs = [[] for _ in range(len(self.raws))]
        self.dropped_windows = [False] * len(self.raws)
        
        if self.do_ica:                
            self.exluded_idxs = [None] * len(self.raws)
            self.labels = [None] * len(self.raws)
            self.y_proba = [None] * len(self.raws)
            
    def get_raws(self):
        return [raw for raw in self.raws if raw is not None]
            
    def run(self):
        total_windows = len(self.raws)
                
        pbar = tqdm(total=total_windows)
        
        for i, raw in enumerate(self.raws):
            start_time = round(self.times[i][0], 1)
            end_time = round(self.times[i][1], 1)
            filename = self.src_paths[self.indices[i]].stem
            
            pbar.set_description(f"File: {filename}.\tTime: {(start_time, end_time)}.")
            
            try:
                self.raws[i] = self.run_single(i)
            except Exception as e:
                print(f"File: {filename}.\tTime: {(start_time, end_time)}.\tError: {e}")
                self.raws[i] = None
                self.dropped_windows[i] = True
                
            pbar.update(1)
    
    def run_single(self, idx: int) -> Optional[mne.io.Raw]:
        raw = self.raws[idx]
               
        quality = self._evaluate_quality(raw)
        if not quality:
            self.dropped_windows[idx] = True
            raw = None
            return None
        
        self._remove_line_noise(raw)
        bad_chs = self._drop_bad_channels(raw)
        self.bad_chs[idx].extend(bad_chs)

        quality = self._evaluate_quality(raw)
        if not quality:
            self.dropped_windows[idx] = True
            raw = None
            return None
        
        self._filter(raw)
        self._average_reference(raw)
                
        if self.do_ica:
            excluded_idxs, labels, y_proba = self._ica_clean(raw)
            self.exluded_idxs[idx] = excluded_idxs
            self.labels[idx] = labels
            self.y_proba[idx] = y_proba
            
            bad_chs = self._drop_bad_channels(raw)
            self.bad_chs[idx].extend(bad_chs)


        missing_chs = self._interpolate_missing(raw)
        self.missing_chs[idx].extend(missing_chs)
        
        extra_chs = self._drop_extra_and_reorder(raw)
        self.extra_chs[idx].extend(extra_chs)
        
        self._interpolate_nearest(raw)
        
        return raw    