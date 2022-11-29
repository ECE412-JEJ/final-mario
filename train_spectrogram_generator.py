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