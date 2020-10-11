import argparse
import glob
import logging
import os
import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from tensorflow_tts.utils import remove_outlier

def generate(data):
    
    tid = data["tid"]
    audio = data["audio"]
    mels = data["mels"]

    samplerate = 16000
    fft_size = 320
    hop_size = 160
    
    # check audio properties
    assert len(audio.shape) == 1, f"{tid} seems to be multi-channel signal."
    assert np.abs(audio).max() <= 1.0, f"{tid} is different from 16 bit PCM."
        
    # get spectrogram
    D = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=None,
        window='hann',
        pad_mode="reflect"
    )
    
    S, _ = librosa.magphase(D)  # (#bins, #frames)
    
    # check audio and feature length
    audio = np.pad(audio, (0, 3200), mode="edge")
    audio = audio[: mels * hop_size]
    assert mels * hop_size == len(audio)
    
    # extract raw pitch
    _f0, t = pw.dio(audio.astype(np.double),fs=samplerate,f0_ceil=8000,frame_period=1000 * hop_size / samplerate)
    f0 = pw.stonemask(audio.astype(np.double), _f0, t, samplerate)
    if len(f0) >= mels:
        f0 = f0[: mels]
    else:
        f0 = np.pad(f0, (0, mels - len(f0)))
        
    # extract energy
    energy = np.sqrt(np.sum(S ** 2, axis=0))
    if len(energy) >= mels:
       energy = energy[: mels]
    else:
       energy = np.pad(energy, (0, mels - len(energy)))
    assert mels == len(f0) == len(energy)
    
    # remove outlier f0/energy
    f0 = remove_outlier(f0)
    energy = remove_outlier(energy)
    
    item = {}
    item["tid"] = tid
    item["f0"] = f0
    item["energy"] = energy
    return item
    
def main():
    parser = argparse.ArgumentParser(description="Dump F0 & Energy")
    parser.add_argument("--outdir", default="./datasets/jsut/basic", type=str, help="directory to save f0 or energy file.")
    parser.add_argument("--rootdir", default="./datasets/jsut/basic", type=str, help="dataset directory root")
    args = parser.parse_args()
    rootdir = args.rootdir
      
    datasets = []
    with open(os.path.join(rootdir, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            tid, _ = line.strip().split("|")
            pcm_path = os.path.join(rootdir, "pcm", f"{tid}.s16")
            feat_path = os.path.join(rootdir, "feats", f"{tid}.f32")
            
            audio, rate = sf.read(pcm_path, samplerate=16000, channels=1, format='RAW', subtype='PCM_16')
            data = {}
            data["tid"] = tid
            data["audio"] = audio
            data["mels"] = os.stat(feat_path).st_size // 4 // 20
            datasets.append(data)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    partial_fn = partial(generate)
    
    p = Pool(4)
    items = p.imap_unordered(partial_fn,tqdm(datasets, total=len(datasets), desc="[Preprocessing]"),chunksize=10)
    
    f0_path = os.path.join(rootdir, "f0")
    if not os.path.exists(f0_path):
        os.makedirs(f0_path)
        
    energy_path = os.path.join(rootdir, "energies")    
    if not os.path.exists(energy_path):
        os.makedirs(energy_path)
        
    for item in items:
        tid = item["tid"]
        f0 = item["f0"]
        energy = item["energy"]
        f0.astype(np.float32).tofile(os.path.join(f0_path, f"{tid}.f0"))
        energy.astype(np.float32).tofile(os.path.join(energy_path, f"{tid}.e"))
            
if __name__ == "__main__":
    main()