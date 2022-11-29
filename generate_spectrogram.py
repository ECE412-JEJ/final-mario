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