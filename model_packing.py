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
