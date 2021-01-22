#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

import numpy as np
import proper 

def mft2( field_in, dout, D, nout, direction, xoffset=0, yoffset=0, xc=0, yc=0 ):

    nfield_in = field_in.shape[1] 
    nfield_out = int(nout)
 
    x = (np.arange(nfield_in) - nfield_in//2 - xc) 
    y = (np.arange(nfield_in) - nfield_in//2 - yc) 

    u = (np.arange(nfield_out) - nfield_out//2 - xoffset/dout) * (dout/D)
    v = (np.arange(nfield_out) - nfield_out//2 - yoffset/dout) * (dout/D)

    xu = np.outer(x, u)
    yv = np.outer(y, v)

    if direction == -1:
        expxu = dout/D * np.exp(-2.0 * np.pi * -1j * xu)
        expyv = np.exp(-2.0 * np.pi * -1j * yv).T
    else:
        expxu = dout/D * np.exp(-2.0 * np.pi * 1j * xu)
        expyv = np.exp(-2.0 * np.pi * 1j * yv).T

    t1 = np.dot(expyv, field_in)
    t2 = np.dot(t1, expxu)

    return t2

def ffts( wavefront, direction ):
    if wavefront.dtype != 'complex128' and wavefront.dtype != 'complex64':
        wavefront = wavefront.astype(complex)

    n = wavefront.shape[0]  # assumed to be power of 2
    wavefront[:,:] = np.roll( np.roll(wavefront, -n//2, 0), -n//2, 1 )  # shift to corner
    
    if proper.use_fftw:
        proper.prop_load_fftw_wisdom( n, proper.fftw_multi_nthreads ) 
        if direction == -1:
            proper.prop_fftw( wavefront, directionFFTW='FFTW_FORWARD' ) 
            wavefront /= np.size(wavefront)
        else:
            proper.prop_fftw( wavefront, directionFFTW='FFTW_BACKWARD' ) 
            wavefront *= np.size(wavefront)
    else:
        if direction == -1:
            wavefront[:,:] = np.fft.fft2(wavefront) / np.size(wavefront)
        else:
            wavefront[:,:] = np.fft.ifft2(wavefront) * np.size(wavefront)
    
    wavefront[:,:] = np.roll( np.roll(wavefront, n//2, 0), n//2, 1 )    # shift to center 

    return wavefront

def pol2rect(amp, phs):
    return amp * np.exp(1j*phs)

def rect2pol(x):
    return abs(x), angle(x)

def trim( input_image, output_dim ):

    input_dim = input_image.shape[1]

    if input_dim == output_dim:
        return input_image
    elif output_dim < input_dim:
        x1 = input_dim // 2 - output_dim // 2
        x2 = x1 + output_dim
        output_image = input_image[x1:x2,x1:x2].copy()
    else:
        output_image = np.zeros((output_dim,output_dim), dtype=input_image.dtype)
        x1 = output_dim // 2 - input_dim // 2
        x2 = x1 + input_dim
        output_image[x1:x2,x1:x2] = input_image

    return output_image




