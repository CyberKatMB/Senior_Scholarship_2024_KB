# Very white because recording taken at night -- darker with recording during the day (but has to be better quality)

from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from PIL import Image
from scipy.io import wavfile
from scipy.signal import hilbert, resample

# APT picture line format
COMPONENT_SIZES = {
    "sync_a": (0, 39),
    "space_a": (39, 86),
    "image_a": (86, 995),
    "telemetry_a": (995, 1040),
    "sync_b": (1040, 1079),
    "space_b": (1079, 1126),
    "image_b": (1126, 2035),
    "telemetry_b": (2035, 2080),
}


SYNCHRONISATION_SEQUENCE = np.array([0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 0, 0, 0,
                                     0, 0, 0]) - 128




def audio_to_hilbert(audio_file):
    """
    1. Load the audio.
    - The audio is a real signal, thus a real function of time.
    - Sound has both real and imaginary components. We take only the real component.
    (the signal is made out of various frequencies with different amplitudes added together.)

    2. Perform Hilbert transform
    - This produces a real signal phase-shifted by 90 degrees for every frequency component of the original real signal
    - (e.g. real signal = sine wave, phase-shifted signal = cosine wave)
    - We then add the original signal and the Hilbert signal together.
    - We say the signal created has a real component (original signal) and imaginary component (Hilbert signal)
    - Using complex notation means examining instantaneous amplitude (envelope) and phase (i.e. examining amplitude and phase at any point in the signal) is easier
      as we can use modulus argument forms
    - e.g. instantaneous amplitude = sqrt((real signal amplitude)^2 + (Hilbert signal amplitude)^2)
    - e.g. instantaneous phase = tan^-1(Hilbert signal/real signal).
    - HOWEVER, we can do this all in one line of code using the SciPy library (thank you to my CS forebears)
    """

    # Rate = 44100Hz (industry standard)
    # Data = audio data of signal
    # Loading audio file
    rate, data = scipy.io.wavfile.read(audio_file)    

    # Resample audio at an appropriate rate
    resample_rate = 20800 # 4160 samples in one line
    coef = resample_rate / rate
    samples = int(coef * len(data))
    data = scipy.signal.resample(data, samples)

    # If there are two channels of signal, average across the channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Hilbert transform audio and extract envelope information
    hilbert_transformed = np.abs(hilbert(data))

    return hilbert_transformed
# ------------------------------------------------------------------------------- #
# Reduces no. of samples that we need to process (reduces time for programme)
def subsample(data):
    step = 5
    return resample(data, len(data) // step)
# ------------------------------------------------------------------------------- #
# Signal to noise ratio (SNR) calculation
def signal_to_noise(data, axis=0):
    # Mean and standard deviation along axis
    mean, std = data.mean(axis), data.std(axis=axis)

    # SNR calculation along axis
    # SNR = mean/std. If std == 0, then set std to 0
    return np.where(std == 0, 0, mean / std)
# ------------------------------------------------------------------------------- #
# Digitise signal to valid pixel intensities from 0-255
# black_point: lower value
# white_point: higher value
def quantise(data, black_point, white_point):

    #  percent for upper and lower saturation
    low, high = np.percentile(data, (black_point, white_point))

    # range adjustment and quantization
    data = np.round((255 * (data - low)) / (high - low)).clip(0, 255)
    # (data-low) shifts data value so low = 0
    # .clip cuts off any values not between 0-255

    # Casts data to 8-bit range
    return data.astype(np.uint8)
# ------------------------------------------------------------------------------- #
# Reshape numpy array into a 2D image array
def reshape(data, synchronisation_sequence: np.ndarray = SYNCHRONISATION_SEQUENCE):
    minimum_row_separation = 2000 # just less than 2080 (one line of one channel's data). Helps to distinguish between rows of data

    # Initialising storage
    rows, previous_corr, previous_ind = [None], -np.inf, 0 # need to have a None value for numpy

    for current_loc in range(len(data) - len(synchronisation_sequence)):
        # Proposed start of row, normalise to zero
        row = [x - 128 for x in data[current_loc : current_loc + len(synchronisation_sequence)]]

        # Correlation between tow and synch sequence
        temp_corr = np.dot(synchronisation_sequence, row)

        # if past the minimum separation, start hunting for next synch
        if current_loc - previous_ind > minimum_row_separation:
            previous_corr, previous_ind = -np.inf, current_loc
            rows.append(data[current_loc : current_loc + 2080])

        # If proposed region matches the sequence better, update
        elif temp_corr > previous_corr:
            previous_corr, previous_ind = temp_corr, current_loc
            rows[-1] = data[current_loc : current_loc + 2080]

    # stack the row to form the image, drop the incomplete rows at the end
    return np.vstack([row for row in rows if len(row) == 2080])
# ------------------------------------------------------------------------------- #
# Filter out noisy rows
def filter_noisy_rows(data):

    # calculate signal to noise and the row to row difference in SNR
    snr = signal_to_noise(data, axis=1)
    snr_diff = np.diff(snr, prepend=0)

    # image filter for rows with high snr (pure noise) and minimal distance
    # in SNR between rows (no theoretical basis, just seems to work)
    data = data[(snr > 0.8) & (snr_diff < 0.05) & (snr_diff > -0.05) & (snr_diff != 0), :]

    return data
# ------------------------------------------------------------------------------- #
# Select components of image to include (see COMPONENTS dictionary)
def select_image_components(data, components: Optional[List[str]]):

    # image array components
    image_regions = []

    # if there are no components, return the full image
    if components is None:
        return data

    # image components to include, based on column down selection
    for component in components:
        component_start, component_end = COMPONENT_SIZES[component]
        image_regions.append(data[:, component_start:component_end])

    return np.hstack(image_regions) # horizontally stacks arrays so concatenates the selected image components side by side
# ------------------------------------------------------------------------------- #
# Save image
def save_image(data, out_path):
    image = Image.fromarray(data.astype(np.uint8))
    image.save(out_path)
# ------------------------------------------------------------------------------- #
# False colourisation based on greyscale intensity
def apply_colormap(data, cm: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("gist_earth")):

    # Get the color map by name:
    colorized = cm(data)

    return colorized[:, :, :3] * 255
    # 3 colour channels (RGB) not 4 as transparency not needed here
    # colour outputs are [0,1] but we need range [0,255] so scale this
# ------------------------------------------------------------------------------- #


def decode_noaa_audio_to_image(in_path, out_path, black_point, white_point, components: Optional[List[str]] = None, colorize: bool = False):

    # components of the image to include
    if components is not None:
        assert set(components) < set(COMPONENT_SIZES.keys()), (
            f"Default to all segments when 'components' is None, otherwise all"
            f"list elements must be in {COMPONENT_SIZES.keys()}"
        )

    # load the audio and convert to Hilbert transformed amplitude info
    decoded = audio_to_hilbert(in_path)
    print(1)

    # sampling from the Hilbert transformed signal for desired images
    subsampled = subsample(decoded)
    print(2)

    # digitize signal to valid pixel intensities in the uint8 range
    quantised = quantise(subsampled, black_point=black_point, white_point=white_point)
    print(3)

    # reshape the numpy array to a 2D image array
    reshaped = reshape(quantised)
    print(4)

    # some empirically based filters for noisy rows
    denoised = filter_noisy_rows(reshaped)
    print(5)

    # select the image components to include
    image_components = select_image_components(denoised, components)
    print(6)

    # colorize greyscale image if selected
    if colorize:
        image_components = apply_colormap(image_components)
    print(7)

    # write numpy array to image file
    save_image(image_components, out_path=out_path)
    print(8)
# ------------------------------------------------------------------------------- #

decode_noaa_audio_to_image(
        in_path="2024-08-23-2134-RAW-NOAA19.wav",
        out_path="2024-08-23-2134-RAW-NOAA19.png",
        black_point=5,
        white_point=95,
        components=["image_a"],
        colorize=False
    )

# if __name__ == "__main__":
#     fire.Fire(decode_noaa_audio_to_image)