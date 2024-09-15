import torch
import numpy as np
import os
import argparse
import utils
import json
from scipy.io.wavfile import write

device = torch.device("cpu")

def main(args, config):
    with torch.no_grad():
        vocoder = torch.jit.load(args.vocoder, map_location=device)
        vocoder.eval()
        mel = torch.from_numpy(np.load(args.mel_path)).to(device)
        mel = mel.unsqueeze(0)
        zero = torch.full((1, 80, 10), -11.5129).to(device=device)
        mels = torch.cat((mel, zero), dim=2)
        wav = vocoder(mels)
        audio = wav.squeeze()
        audio = audio[: -(config.hop_length * 10)]
        audio = audio.cpu().float().numpy()
        audio = audio / (audio.max() - audio.min())
    save_path = os.path.join(args.save_path, f"{config.dataset}_synth.wav")
    write(save_path, config.sampling_rate, audio)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder", type=str, required=True,
        help="Path to the pretrained vocoder model")
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--mel_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    main(args, config)

