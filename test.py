import torch, torchaudio
from mdct import MDCT


def main():

    path = 'samples/original.wav'
    wav, sr = torchaudio.load(path)

    filter_length = 1440
    window_length = 480

    mdct = MDCT(filter_length, window_length)

    rec = mdct.reconstruct(wav)
    torchaudio.save('samples/reconstructed.wav', src=rec, sample_rate=sr)

    return 0


if __name__ == "__main__":
    main()
