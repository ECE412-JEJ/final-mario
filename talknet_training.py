import os

dataset = "LJSpeech-1.1/datasets.zip" #@param {type:"string"}
train_filelist = "LJSpeech-1.1/train_filelist.txt" #@param {type:"string"}
val_filelist = "LJSpeech-1.1/val_filelist.txt" #@param {type:"string"}
output_dir = "gen_output" #@param {type:"string"}
assert os.path.exists(dataset), "Cannot find dataset"
assert os.path.exists(train_filelist), "Cannot find training filelist"
assert os.path.exists(val_filelist), "Cannot find validation filelist"
if not os.path.exists(output_dir):
   os.makedirs(output_dir)
print("OK")

import time
import gdown
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
import ffmpeg

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
os.chdir('/content')
if os.path.exists("/content/wavs"):
    shutil.rmtree("/content/wavs")
os.mkdir("wavs")
os.chdir("wavs")
if dataset[-4:] == ".zip":
    !unzip -q "{dataset}"
elif dataset[-4:] == ".tar":
    !tar -xf "{dataset}"
else:
    raise Exception("Unknown extension for dataset")
if os.path.exists("/content/wavs/wavs"):
    shutil.move("/content/wavs/wavs", "/content/tempwavs")
    shutil.rmtree("/content/wavs")
    shutil.move("/content/tempwavs", "/content/wavs")

# Filelist for preprocessing
os.chdir('/content')
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
with open("/content/allfiles.txt", "w") as f:
    for x in all_filelist:
        if x.strip() == "":
            continue
        if x.split("|")[0] not in seen_files:
            seen_files.append(x.split("|")[0])
            f.write(x.strip() + "\n")

# Ensure audio is 22k
print("Converting audio...")
for r, _, f in os.walk("/content/wavs"):
    for name in tqdm(f):
        convert_to_22k(os.path.join(r, name))

# Convert to JSON
generate_json("trainfiles.txt", "trainfiles.json")
generate_json("valfiles.txt", "valfiles.json")
generate_json("allfiles.txt", "allfiles.json")

print("OK")

#@markdown **Step 6:** Dataset processing, part 2. This takes a while, but
#@markdown you only have to run this once per dataset (results are saved to Drive).

#@markdown If this step fails, try the following:
#@markdown * Make sure your dataset only contains WAV files.

# Extract phoneme duration

import json
from nemo.collections.asr.models import EncDecCTCModel
asr_model = EncDecCTCModel.from_pretrained(model_name="asr_talknet_aligner").cpu().eval()

def forward_extractor(tokens, log_probs, blank):
    """Computes states f and p."""
    n, m = len(tokens), log_probs.shape[0]
    # `f[s, t]` -- max sum of log probs for `s` first codes
    # with `t` first timesteps with ending in `tokens[s]`.
    f = np.empty((n + 1, m + 1), dtype=float)
    f.fill(-(10 ** 9))
    p = np.empty((n + 1, m + 1), dtype=int)
    f[0, 0] = 0.0  # Start
    for s in range(1, n + 1):
        c = tokens[s - 1]
        for t in range((s + 1) // 2, m + 1):
            f[s, t] = log_probs[t - 1, c]
            # Option #1: prev char is equal to current one.
            if s == 1 or c == blank or c == tokens[s - 3]:
                options = f[s : (s - 2 if s > 1 else None) : -1, t - 1]
            else:  # Is not equal to current one.
                options = f[s : (s - 3 if s > 2 else None) : -1, t - 1]
            f[s, t] += np.max(options)
            p[s, t] = np.argmax(options)
    return f, p


def backward_extractor(f, p):
    """Computes durs from f and p."""
    n, m = f.shape
    n -= 1
    m -= 1
    durs = np.zeros(n, dtype=int)
    if f[-1, -1] >= f[-2, -1]:
        s, t = n, m
    else:
        s, t = n - 1, m
    while s > 0:
        durs[s - 1] += 1
        s -= p[s, t]
        t -= 1
    assert durs.shape[0] == n
    assert np.sum(durs) == m
    assert np.all(durs[1::2] > 0)
    return durs

def preprocess_tokens(tokens, blank):
    new_tokens = [blank]
    for c in tokens:
        new_tokens.extend([c, blank])
    tokens = new_tokens
    return tokens

data_config = {
    'manifest_filepath': "allfiles.json",
    'sample_rate': 22050,
    'labels': asr_model.decoder.vocabulary,
    'batch_size': 1,
}

parser = nemo.collections.asr.data.audio_to_text.AudioToCharWithDursF0Dataset.make_vocab(
    notation='phonemes', punct=True, spaces=True, stresses=False, add_blank_at="last"
)

dataset = nemo.collections.asr.data.audio_to_text._AudioTextDataset(
    manifest_filepath=data_config['manifest_filepath'], sample_rate=data_config['sample_rate'], parser=parser,
)

dl = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=data_config['batch_size'], collate_fn=dataset.collate_fn, shuffle=False,
)

blank_id = asr_model.decoder.num_classes_with_blank - 1

if os.path.exists(os.path.join(output_dir, "durations.pt")):
    print("durations.pt already exists; skipping")
else:
    dur_data = {}
    for sample_idx, test_sample in tqdm(enumerate(dl), total=len(dl)):
        log_probs, _, greedy_predictions = asr_model(
            input_signal=test_sample[0], input_signal_length=test_sample[1]
        )

        log_probs = log_probs[0].cpu().detach().numpy()
        seq_ids = test_sample[2][0].cpu().detach().numpy()

        target_tokens = preprocess_tokens(seq_ids, blank_id)

        f, p = forward_extractor(target_tokens, log_probs, blank_id)
        durs = backward_extractor(f, p)

        dur_key = Path(dl.dataset.collection[sample_idx].audio_file).stem
        dur_data[dur_key] = {
            'blanks': torch.tensor(durs[::2], dtype=torch.long).cpu().detach(), 
            'tokens': torch.tensor(durs[1::2], dtype=torch.long).cpu().detach()
        }

        del test_sample

    torch.save(dur_data, os.path.join(output_dir, "durations.pt"))

#Extract F0 (pitch)
import crepe
from scipy.io import wavfile

def crepe_f0(audio_file, hop_length=256):
    sr, audio = wavfile.read(audio_file)
    audio_x = np.arange(0, len(audio)) / 22050.0
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    x = np.arange(0, len(audio), hop_length) / 22050.0
    freq_interp = np.interp(x, time, frequency)
    conf_interp = np.interp(x, time, confidence)
    audio_interp = np.interp(x, audio_x, np.absolute(audio)) / 32768.0
    weights = [0.5, 0.25, 0.25]
    audio_smooth = np.convolve(audio_interp, np.array(weights)[::-1], "same")

    conf_threshold = 0.25
    audio_threshold = 0.0005
    for i in range(len(freq_interp)):
        if conf_interp[i] < conf_threshold:
            freq_interp[i] = 0.0
        if audio_smooth[i] < audio_threshold:
            freq_interp[i] = 0.0

    # Hack to make f0 and mel lengths equal
    if len(audio) % hop_length == 0:
        freq_interp = np.pad(freq_interp, pad_width=[0, 1])
    return torch.from_numpy(freq_interp.astype(np.float32))

if os.path.exists(os.path.join(output_dir, "f0s.pt")):
    print("f0s.pt already exists; skipping")
else:
    f0_data = {}
    with open("allfiles.json") as f:
        for i, l in enumerate(f.readlines()):
            print(str(i))
            audio_path = json.loads(l)["audio_filepath"]
            f0_data[Path(audio_path).stem] = crepe_f0(audio_path)

    # calculate f0 stats (mean & std) only for train set
    with open("trainfiles.json") as f:
        train_ids = {Path(json.loads(l)["audio_filepath"]).stem for l in f}
    all_f0 = torch.cat([f0[f0 >= 1e-5] for f0_id, f0 in f0_data.items() if f0_id in train_ids])

    F0_MEAN, F0_STD = all_f0.mean().item(), all_f0.std().item()        
    print("F0_MEAN: " + str(F0_MEAN) + ", F0_STD: " + str(F0_STD))
    torch.save(f0_data, os.path.join(output_dir, "f0s.pt"))
    with open(os.path.join(output_dir, "f0_info.json"), "w") as f:
        f.write(json.dumps({"FO_MEAN": F0_MEAN, "F0_STD": F0_STD}))

#@markdown **Step 7:** Train duration predictor.

#@markdown If CUDA runs out of memory, try the following:
#@markdown * Click on Runtime -> Restart runtime, re-run step 3, and try again.
#@markdown * If that doesn't help, reduce the batch size (default 64).
batch_size = 64 #@param {type:"integer"}

epochs = 20
learning_rate = 1e-3
min_learning_rate = 3e-6
load_checkpoints = True

import os
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import TalkNetDursModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

def train(cfg):
    cfg.sample_rate = 22050
    cfg.train_dataset = "trainfiles.json"
    cfg.validation_datasets = "valfiles.json"
    cfg.durs_file = os.path.join(output_dir, "durations.pt")
    cfg.f0_file = os.path.join(output_dir, "f0s.pt")
    cfg.trainer.accelerator = "dp"
    cfg.trainer.max_epochs = epochs
    cfg.trainer.check_val_every_n_epoch = 5
    cfg.model.train_ds.dataloader_params.batch_size = batch_size
    cfg.model.validation_ds.dataloader_params.batch_size = batch_size
    cfg.model.optim.lr = learning_rate
    cfg.model.optim.sched.min_lr = min_learning_rate
    cfg.exp_manager.exp_dir = output_dir

    # Find checkpoints
    ckpt_path = ""
    if load_checkpoints:
      path0 = os.path.join(output_dir, "TalkNetDurs")
      if os.path.exists(path0):
          path1 = sorted(os.listdir(path0))
          for i in range(len(path1)):
              path2 = os.path.join(path0, path1[-(1+i)], "checkpoints")
              if os.path.exists(path2):
                  match = [x for x in os.listdir(path2) if "last.ckpt" in x]
                  if len(match) > 0:
                      ckpt_path = os.path.join(path2, match[0])
                      print("Resuming training from " + match[0])
                      break
    
    if ckpt_path != "":
        trainer = pl.Trainer(**cfg.trainer, resume_from_checkpoint = ckpt_path)
        model = TalkNetDursModel(cfg=cfg.model, trainer=trainer)
    else:
        warmstart_path = "/content/talknet_durs.nemo"
        trainer = pl.Trainer(**cfg.trainer)
        model = TalkNetDursModel.restore_from(warmstart_path, override_config_path=cfg)
        model.set_trainer(trainer)
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        model.setup_optimization(cfg.model.optim)
        print("Warm-starting from " + warmstart_path)
    exp_manager(trainer, cfg.get('exp_manager', None))
    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()])  # noqa
    trainer.fit(model)

GlobalHydra().clear()
initialize(config_path="conf")
cfg = compose(config_name="talknet-durs")
train(cfg)

#@markdown **Step 8:** Train pitch predictor.

#@markdown If CUDA runs out of memory, try the following:
#@markdown * Click on Runtime -> Restart runtime, re-run step 3, and try again.
#@markdown * If that doesn't help, reduce the batch size (default 64).
batch_size = 64 #@param {type:"integer"}
epochs = 50

import json

with open(os.path.join(output_dir, "f0_info.json"), "r") as f:
    f0_info = json.load(f)
    f0_mean = f0_info["FO_MEAN"]
    f0_std = f0_info["F0_STD"]

learning_rate = 1e-3
min_learning_rate = 3e-6
load_checkpoints = True

import os
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import TalkNetPitchModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

def train(cfg):
    cfg.sample_rate = 22050
    cfg.train_dataset = "trainfiles.json"
    cfg.validation_datasets = "valfiles.json"
    cfg.durs_file = os.path.join(output_dir, "durations.pt")
    cfg.f0_file = os.path.join(output_dir, "f0s.pt")
    cfg.trainer.accelerator = "dp"
    cfg.trainer.max_epochs = epochs
    cfg.trainer.check_val_every_n_epoch = 5
    cfg.model.f0_mean=f0_mean
    cfg.model.f0_std=f0_std
    cfg.model.train_ds.dataloader_params.batch_size = batch_size
    cfg.model.validation_ds.dataloader_params.batch_size = batch_size
    cfg.model.optim.lr = learning_rate
    cfg.model.optim.sched.min_lr = min_learning_rate
    cfg.exp_manager.exp_dir = output_dir

    # Find checkpoints
    ckpt_path = ""
    if load_checkpoints:
      path0 = os.path.join(output_dir, "TalkNetPitch")
      if os.path.exists(path0):
          path1 = sorted(os.listdir(path0))
          for i in range(len(path1)):
              path2 = os.path.join(path0, path1[-(1+i)], "checkpoints")
              if os.path.exists(path2):
                  match = [x for x in os.listdir(path2) if "last.ckpt" in x]
                  if len(match) > 0:
                      ckpt_path = os.path.join(path2, match[0])
                      print("Resuming training from " + match[0])
                      break
    
    if ckpt_path != "":
        trainer = pl.Trainer(**cfg.trainer, resume_from_checkpoint = ckpt_path)
        model = TalkNetPitchModel(cfg=cfg.model, trainer=trainer)
    else:
        warmstart_path = "/content/talknet_pitch.nemo"
        trainer = pl.Trainer(**cfg.trainer)
        model = TalkNetPitchModel.restore_from(warmstart_path, override_config_path=cfg)
        model.set_trainer(trainer)
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        model.setup_optimization(cfg.model.optim)
        print("Warm-starting from " + warmstart_path)
    exp_manager(trainer, cfg.get('exp_manager', None))
    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()])  # noqa
    trainer.fit(model)

GlobalHydra().clear()
initialize(config_path="conf")
cfg = compose(config_name="talknet-pitch")
train(cfg)

#@markdown **Step 9:** Train spectrogram generator. 200+ epochs are recommended. 

#@markdown This is the slowest of the three models to train, and the hardest to
#@markdown get good results from. If your character sounds noisy or robotic,
#@markdown try improving the dataset, or adjusting the epochs and learning rate.

epochs = 200 #@param {type:"integer"}

#@markdown If CUDA runs out of memory, try the following:
#@markdown * Click on Runtime -> Restart runtime, re-run step 3, and try again.
#@markdown * If that doesn't help, reduce the batch size (default 32).
batch_size = 32 #@param {type:"integer"}

#@markdown Advanced settings. You can probably leave these at their defaults (1e-3, 3e-6, empty, checked).
learning_rate = 1e-3 #@param {type:"number"}
min_learning_rate = 3e-6 #@param {type:"number"}
pretrained_path = "" #@param {type:"string"}
load_checkpoints = True #@param {type:"boolean"}

import os
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import TalkNetSpectModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

def train(cfg):
    cfg.sample_rate = 22050
    cfg.train_dataset = "trainfiles.json"
    cfg.validation_datasets = "valfiles.json"
    cfg.durs_file = os.path.join(output_dir, "durations.pt")
    cfg.f0_file = os.path.join(output_dir, "f0s.pt")
    cfg.trainer.accelerator = "dp"
    cfg.trainer.max_epochs = epochs
    cfg.trainer.check_val_every_n_epoch = 5
    cfg.model.train_ds.dataloader_params.batch_size = batch_size
    cfg.model.validation_ds.dataloader_params.batch_size = batch_size
    cfg.model.optim.lr = learning_rate
    cfg.model.optim.sched.min_lr = min_learning_rate
    cfg.exp_manager.exp_dir = output_dir

    # Find checkpoints
    ckpt_path = ""
    if load_checkpoints:
      path0 = os.path.join(output_dir, "TalkNetSpect")
      if os.path.exists(path0):
          path1 = sorted(os.listdir(path0))
          for i in range(len(path1)):
              path2 = os.path.join(path0, path1[-(1+i)], "checkpoints")
              if os.path.exists(path2):
                  match = [x for x in os.listdir(path2) if "last.ckpt" in x]
                  if len(match) > 0:
                      ckpt_path = os.path.join(path2, match[0])
                      print("Resuming training from " + match[0])
                      break
    
    if ckpt_path != "":
        trainer = pl.Trainer(**cfg.trainer, resume_from_checkpoint = ckpt_path)
        model = TalkNetSpectModel(cfg=cfg.model, trainer=trainer)
    else:
        if pretrained_path != "":
            warmstart_path = pretrained_path
        else:
            warmstart_path = "/content/talknet_spect.nemo"
        trainer = pl.Trainer(**cfg.trainer)
        model = TalkNetSpectModel.restore_from(warmstart_path, override_config_path=cfg)
        model.set_trainer(trainer)
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        model.setup_optimization(cfg.model.optim)
        print("Warm-starting from " + warmstart_path)
    exp_manager(trainer, cfg.get('exp_manager', None))
    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()])  # noqa
    trainer.fit(model)

GlobalHydra().clear()
initialize(config_path="conf")
cfg = compose(config_name="talknet-spect")
train(cfg)

#@markdown **Step 10:** Generate GTA spectrograms. This will help HiFi-GAN learn what your TalkNet model sounds like.

#@markdown If this step fails, make sure you've finished training the spectrogram generator.

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from nemo.collections.tts.models import TalkNetSpectModel
import shutil

def fix_paths(inpath):
    output = ""
    with open(inpath, "r", encoding="utf8") as f:
        for l in f.readlines():
            if l[:5].lower() != "wavs/":
                output += "wavs/" + l
            else:
                output += l
    with open(inpath, "w", encoding="utf8") as w:
        w.write(output)

shutil.copyfile(train_filelist, "/content/hifi-gan/training.txt")
shutil.copyfile(val_filelist, "/content/hifi-gan/validation.txt")
fix_paths("/content/hifi-gan/training.txt")
fix_paths("/content/hifi-gan/validation.txt")
fix_paths("/content/allfiles.txt")

os.chdir('/content')
indir = "wavs"
outdir = "hifi-gan/wavs"
if not os.path.exists(outdir):
    os.mkdir(outdir)

model_path = ""
path0 = os.path.join(output_dir, "TalkNetSpect")
if os.path.exists(path0):
    path1 = sorted(os.listdir(path0))
    for i in range(len(path1)):
        path2 = os.path.join(path0, path1[-(1+i)], "checkpoints")
        if os.path.exists(path2):
            match = [x for x in os.listdir(path2) if "TalkNetSpect.nemo" in x]
            if len(match) > 0:
                model_path = os.path.join(path2, match[0])
                break
assert model_path != "", "TalkNetSpect.nemo not found"

dur_path = os.path.join(output_dir, "durations.pt")
f0_path = os.path.join(output_dir, "f0s.pt")

model = TalkNetSpectModel.restore_from(model_path)
model.eval()
with open("allfiles.txt", "r", encoding="utf-8") as f:
    dataset = f.readlines()
durs = torch.load(dur_path)
f0s = torch.load(f0_path)

for x in tqdm(dataset):
    x_name = os.path.splitext(os.path.basename(x.split("|")[0].strip()))[0]
    x_tokens = model.parse(text=x.split("|")[1].strip())
    x_durs = (
        torch.stack(
            (
                durs[x_name]["blanks"],
                torch.cat((durs[x_name]["tokens"], torch.zeros(1).int())),
            ),
            dim=1,
        )
        .view(-1)[:-1]
        .view(1, -1)
        .to("cuda:0")
    )
    x_f0s = f0s[x_name].view(1, -1).to("cuda:0")
    x_spect = model.force_spectrogram(tokens=x_tokens, durs=x_durs, f0=x_f0s)
    rel_path = os.path.splitext(x.split("|")[0].strip())[0][5:]
    abs_dir = os.path.join(outdir, os.path.dirname(rel_path))
    if abs_dir != "" and not os.path.exists(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)
    np.save(os.path.join(outdir, rel_path + ".npy"), x_spect.detach().cpu().numpy())

#@markdown **Step 11:** Train HiFi-GAN. 2,000+ steps are recommended.
#@markdown Stop this cell to finish training the model.

#@markdown If CUDA runs out of memory, click on Runtime -> Restart runtime, re-run step 3, and try again.
#@markdown If this step still fails to start, make sure step 10 finished successfully.

#@markdown Note: If the training process starts at step 2500000, delete the HiFiGAN folder and try again.

import gdown
d = 'https://drive.google.com/uc?id='

os.chdir('/content/hifi-gan')
assert os.path.exists("wavs"), "Spectrogram folder not found"

if not os.path.exists(os.path.join(output_dir, "HiFiGAN")):
    os.makedirs(os.path.join(output_dir, "HiFiGAN"))
if not os.path.exists(os.path.join(output_dir, "HiFiGAN", "do_00000000")):
    print("Downloading universal model...")
    gdown.download(d+"1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW", os.path.join(output_dir, "HiFiGAN", "g_00000000"), quiet=False)
    gdown.download(d+"1O63eHZR9t1haCdRHQcEgMfMNxiOciSru", os.path.join(output_dir, "HiFiGAN", "do_00000000"), quiet=False)
    start_from_universal = "--warm_start True "
else:
    start_from_universal = ""

!python train.py --fine_tuning True --config config_v1b.json \
{start_from_universal} \
--checkpoint_interval 250 --checkpoint_path "{os.path.join(output_dir, 'HiFiGAN')}" \
--input_training_file "/content/hifi-gan/training.txt" \
--input_validation_file "/content/hifi-gan/validation.txt" \
--input_wavs_dir ".." --input_mels_dir "wavs"

#@markdown **Step 12:** Package the models. They'll be saved to the output directory as [character_name]_TalkNet.zip.

character_name = "Character" #@param {type:"string"}

#@markdown When done, generate a Drive share link, with permissions set to "Anyone with the link". 
#@markdown You can then use it with the [Controllable TalkNet notebook](https://colab.research.google.com/drive/1aj6Jk8cpRw7SsN3JSYCv57CrR6s0gYPB) 
#@markdown by selecting "Custom model" as your character.

#@markdown This cell will also move the training checkpoints and logs to the trash.
#@markdown That should free up roughly 2 GB of space on your Drive (remember to empty your trash).
#@markdown If you wish to keep them, uncheck this box.

delete_checkpoints = True #@param {type:"boolean"}

import shutil
from zipfile import ZipFile

def find_talknet(model_dir):
    ckpt_path = ""
    path0 = os.path.join(output_dir, model_dir)
    if os.path.exists(path0):
        path1 = sorted(os.listdir(path0))
        for i in range(len(path1)):
            path2 = os.path.join(path0, path1[-(1+i)], "checkpoints")
            if os.path.exists(path2):
                match = [x for x in os.listdir(path2) if ".nemo" in x]
                if len(match) > 0:
                    ckpt_path = os.path.join(path2, match[0])
                    break
    assert ckpt_path != "", "Couldn't find " + model_dir
    return ckpt_path

durs_path = find_talknet("TalkNetDurs")
pitch_path = find_talknet("TalkNetPitch")
spect_path = find_talknet("TalkNetSpect")
assert os.path.exists(os.path.join(output_dir, "HiFiGAN", "g_00000000")), "Couldn't find HiFi-GAN"

zip = ZipFile(os.path.join(output_dir, character_name + "_TalkNet.zip"), 'w')
zip.write(durs_path, "TalkNetDurs.nemo")
zip.write(pitch_path, "TalkNetPitch.nemo")
zip.write(spect_path, "TalkNetSpect.nemo")
zip.write(os.path.join(output_dir, "HiFiGAN", "g_00000000"), "hifiganmodel")
zip.write(os.path.join(output_dir, "HiFiGAN", "config.json"), "config.json")
zip.write(os.path.join(output_dir, "f0_info.json"), "f0_info.json")
zip.close()
print("Archived model to " + os.path.join(output_dir, character_name + "_TalkNet.zip"))

if delete_checkpoints:
    shutil.rmtree((os.path.join(output_dir, "TalkNetDurs")))
    shutil.rmtree((os.path.join(output_dir, "TalkNetPitch")))
    shutil.rmtree((os.path.join(output_dir, "TalkNetSpect")))
    shutil.rmtree((os.path.join(output_dir, "HiFiGAN")))
