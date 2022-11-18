import json
from nemo.collections.asr.models import EncDecCTCModel

asr_model = EncDecCTCModel.from_pretrained(model_name="asr_talknet_aligner").cpu().eval()

def forward_extractor(tokens, log_probs, blank):
    n, m = len(tokens), log_probs.shape[0]

