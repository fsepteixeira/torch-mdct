# torch-mdct
Pytorch implementation of the 1D Modified Discrete Cosine Transform (MDCT) and its inverse (iMDCT).

The MDCT and its inverse are implemented using 1D convolutions and 1D transpose convolutions respectively, using a filterbank of Kaiser-Bessel Derived windows with alpha = 4.

The current implementation only allows even filter_lengths and always uses 50% overlap to guarantee perfect reconstruction. 
This module is an adaptation/combination of the following github repositories:

-- https://github.com/nils-werner/mdct
-- https://github.com/pseeth/torch-stft

It follows the MDCT implementation described by Bosi, Marina and Richard E. Goldberg in:
-- Introduction to digital audio coding and standards, 
   Vol. 721. Springer Science & Business Media, 2012,
   https://www.springer.com/gp/book/9781402073571.
