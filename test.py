import torch, torchaudio
from mdct import MDCT


def main():

    path = 'samples/original.wav'
    wav, sr = torchaudio.load(path)

    filter_length = int(30e-3*sr)
    window_length = int(10e-3*sr)

    mdct = MDCT(filter_length, window_length, pad=True, save_pad=True)

    rec = mdct.reconstruct(wav)
    torchaudio.save('samples/reconstructed.wav', src=rec, sample_rate=sr)

    return 0


if __name__ == "__main__":
    main()
