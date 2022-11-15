import numpy as np

import torch
import torch.nn.functional as F
from .utils import *

class MDCT(torch.nn.Module):
    def __init__(self, filter_length=1024, window_length=None, pad=False, save_pad=False):
        """
        This module implements the 1D MDCT and its inverse using 1D convolution and 1D transpose convolutions.
        It only allows even filter_lengths and always uses 50% overlap to guarantee perfect 
        reconstruction. This module uses Kaiser-Bessel Derived windows with alpha = 4.
        This module is an adaptation/combination of the following github repositories:
            -- https://github.com/nils-werner/mdct
            -- https://github.com/pseeth/torch-stft
        It follows the MDCT implementation described by Bosi, Marina and Richard E. Goldberg in:
            -- Introduction to digital audio coding and standards, 
                Vol. 721. Springer Science & Business Media, 2012,
                https://www.springer.com/gp/book/9781402073571.

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            win_length {int} -- Length of the window function applied to each frame, 
                should be smaller than the filter_length (if not specified, it
                equals the filter length). (default: {None})
            pad {bool} -- Whether to pad input or not. If True will pad input such that:
                             num_samples % filter_length == 0. 
                          This guarantees that no samples are lost. 
                          If False, the output of the iMDCT will correspond to:
                             num_samples - (num_samples % filter_length) 
                          (equivalent to 'valid' convolution). (default {False})
           save_pad {bool} -- If True, the object will save the last value used for padding and will remove it 
                                after the iMDCT is applied. (default {False})
        """

        super(MDCT, self).__init__()

        self.pad = pad
        self.save_pad = save_pad
        self.filter_length = filter_length
        assert ((filter_length % 2) == 0)

        self.window_length = window_length if window_length is not None else filter_length
        self.hop_size = filter_length // 2

        # get window and zero center pad it to filter_length
        assert (filter_length >= self.window_length)
        self.window = kbd_window_(self.window_length, self.filter_length, alpha=4)

        forward_basis = mdct_basis_(filter_length)
        forward_basis *= self.window.T
        inverse_basis = forward_basis.T

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def mdct(self, input_data):
        """Transform input data (audio) to MDCT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of MDCT with shape (num_batch, 
                num_frequencies, num_frames)
        """

        # Pad data with win_len / 2 on either side
        num_batches, num_samples = input_data.size()
        input_data = input_data.view(num_batches, 1, num_samples)

        if self.pad:
            pad = (num_samples % self.filter_length)
            input_data = F.pad(input_data, (pad, pad, 0, 0), mode='constant')

            if self.save_pad:
                self.pad = pad

        output = F.conv1d(input_data,
                          self.forward_basis.unsqueeze(dim=1),
                          stride=self.hop_size, padding=0)

        # Return magnitude -> MDCT only includes real values
        return output

    def imdct(self, magnitude):
        """Call the inverse MDCT (iMDCT), given the tensor produced 
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of MDCT with shape (num_batch, 
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude. Of
                shape (num_batch, num_samples)
        """
        inverse_transform = F.conv_transpose1d(magnitude,
                                               self.inverse_basis.unsqueeze(dim=1).T,
                                               stride=self.hop_size, padding=0)

        if self.pad and self.save_pad:
            inverse_transform = inverse_transform[..., self.pad:-self.pad]

        return inverse_transform.squeeze(1) * (4 / self.filter_length)

    def reconstruct(self, input_data):
        """Takes input data (audio) to DCT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude. Of
                shape (num_batch, num_samples)
        """
        magnitude = self.mdct(input_data)
        reconstruction = self.imdct(magnitude)
        return reconstruction
