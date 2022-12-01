import os

dataset = "./content/mario-voice-dataset/mario-dataset.zip"
train_filelist = "./content/mario-voice-dataset/train_dataset.txt"
val_filelist = "./content/mario-voice-dataset/val_dataset.txt"
output_dir = "./voice_sample/mario"

assert os.path.exists(dataset), "Cannot find dataset"
assert os.path.exists(train_filelist), "Cannot find training filelist"
assert os.path.exists(val_filelist), "Cannot find validation filelist"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("OK")
