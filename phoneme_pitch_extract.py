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