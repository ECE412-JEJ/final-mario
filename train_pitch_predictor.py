default_path  = "/afs/ee.cooper.edu/user/j/jiyoon.pyo/final-mario"
output_dir = default_path + "/voice_sample/mario/"

batch_size = 64
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
    cfg.train_dataset = default_path + "/content/trainfiles.json"
    cfg.validation_datasets = default_path + "/content/valfiles.json"
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
        warmstart_path = default_path + "/conf/talknet_pitch.nemo"
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