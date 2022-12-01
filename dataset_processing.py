import os
import shutil
import sys
import json
import nemo
import torch
import torchaudio
import numpy as np
from pysptk import sptk
from pathlib import Path
from tqdm.notebook import tqdm
from zipfile import ZipFile
import ffmpeg

default_path  = "/afs/ee.cooper.edu/user/j/jiyoon.pyo/final-mario"
dataset = default_path + "/content/mario-voice-dataset/mario-dataset.zip"
train_filelist = default_path + "/content/mario-voice-dataset/train_dataset.txt"
val_filelist = default_path + "/content/mario-voice-dataset/val_dataset.txt"
output_dir = default_path + "/voice_sample/mario"

def fix_transcripts(inpath):
    found_arpabet = False
    found_grapheme = False
    with open(inpath, "r", encoding="utf8") as f:
        lines = f.readlines()
    with open(inpath, "w", encoding="utf8") as f:
        for l in lines:
            if l.strip() == "":
                continue
            if "{" in l:
                if not found_arpabet:
                    print("Warning: Skipping ARPABET lines (not supported).")
                    found_arpabet = True
            else:
                f.write(l)
                found_grapheme = True
    assert found_grapheme, "No non-ARPABET lines found in " + inpath

def generate_json(inpath, outpath):
    output = ""
    sample_rate = 22050
    with open(inpath, "r", encoding="utf8") as f:
        for l in f.readlines():
            lpath = l.split("|")[0].strip()
            if lpath[:5] != "wavs/":
                lpath = "wavs/" + lpath
            size = os.stat(
                os.path.join(os.path.dirname(inpath), lpath)
            ).st_size
            x = {
                "audio_filepath": lpath,
                "duration": size / (sample_rate * 2),
                "text": l.split("|")[1].strip(),
            }
            output += json.dumps(x) + "\n"
        with open(outpath, "w", encoding="utf8") as w:
            w.write(output)

def convert_to_22k(inpath):
    if inpath.strip()[-4:].lower() != ".wav":
        print("Warning: " + inpath.strip() + " is not a .wav file!")
        return
    ffmpeg.input(inpath).output(
        inpath + "_22k.wav",
        ar="22050",
        ac="1",
        acodec="pcm_s16le",
        map_metadata="-1",
        fflags="+bitexact",
    ).overwrite_output().run(quiet=True)
    os.remove(inpath)
    os.rename(inpath + "_22k.wav", inpath)

# Extract dataset
os.chdir(default_path + '/content')
if os.path.exists(default_path + "/content/wavs"):
    shutil.rmtree(default_path + "/content/wavs")
os.mkdir("wavs")
os.chdir("wavs")
if dataset[-4:] == ".zip":
    with ZipFile(dataset, 'r') as zObject:
        zObject.extractall(default_path + "/content/wavs")
elif dataset[-4:] == ".tar":
    shutil.unpack_archive(dataset, default_path + "/content/wavs")
else:
    raise Exception("Unknown extension for dataset")
if os.path.exists(default_path + "/content/wavs/wavs"):
    shutil.move(default_path + "/content/wavs/wavs", default_path + "/content/tempwavs")
    shutil.rmtree(default_path + "/content/wavs")
    shutil.move(default_path + "/content/tempwavs", default_path + "/content/wavs")

# Filelist for preprocessing
os.chdir(default_path + '/content')
shutil.copy(train_filelist, "trainfiles.txt")
shutil.copy(val_filelist, "valfiles.txt")
fix_transcripts("trainfiles.txt")
fix_transcripts("valfiles.txt")
seen_files = []
with open("trainfiles.txt") as f:
    t = f.read().split("\n")
with open("valfiles.txt") as f:
    v = f.read().split("\n")
    all_filelist = t[:] + v[:]
with open(default_path + "/content/allfiles.txt", "w") as f:
    for x in all_filelist:
        if x.strip() == "":
            continue
        if x.split("|")[0] not in seen_files:
            seen_files.append(x.split("|")[0])
            f.write(x.strip() + "\n")

# Ensure audio is 22k
print("Converting audio...")
for r, _, f in os.walk(default_path + "/content/wavs"):
    for name in tqdm(f):
        convert_to_22k(os.path.join(r, name))

# Convert to JSON
generate_json("trainfiles.txt", "trainfiles.json")
generate_json("valfiles.txt", "valfiles.json")
generate_json("allfiles.txt", "allfiles.json")

print("OK")
