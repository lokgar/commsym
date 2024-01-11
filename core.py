import numpy as np


def generate_tone(frequency, amplitude, duration, sampling_rate):
    """
    Returns a sinusoidal signal.

    Parameters:
    ----------
    frequency : float
        The frequency of the sinusoidal tone in Hertz (Hz).

    amplitude : float
        The amplitude of the sinusoidal tone.

    duration : float
        The duration of the tone in seconds (s).

    sampling_rate : float
        The sampling rate in samples per second, dictating the resolution of the generated tone.

    Returns:
    --------
    time : np.array
        An array representing the time intervals at which the tone is sampled.

    tone : np.array
        The sinusoidal tone signal as an array, corresponding to the time array.
    """

    time = np.linspace(0, duration, int(duration * sampling_rate))
    tone = amplitude * np.cos(2 * np.pi * frequency * time)
    return time, tone


def generate_pulse_train(
    pulse_rate, pulse_amplitude, fill_factor, duration, sampling_rate
):
    """
    Returns a pulse train.

    PParameters:
    ----------
    pulse_rate : float
        The rate at which pulses occur in the train, in pulses per second.

    pulse_amplitude : float
        The amplitude of each pulse in the train.

    fill_factor : float
        The ratio of the pulse duration to the pulse period (pulse duration / pulse period).

    duration : float
        The total duration of the pulse train in seconds (s).

    sampling_rate : float
        The sampling rate in samples per second, dictating the resolution of the pulse train.

    Returns:
    --------
    time : np.array
        An array representing the time intervals at which the pulse train is sampled.

    pulse_train : np.array
        The generated pulse train signal as an array, corresponding to the time array.
    """

    # Calculate pulse duration based on the filling factor
    pulse_period = 1 / pulse_rate
    pulse_duration = pulse_period * fill_factor

    # Create a time vector
    time = np.linspace(0, duration, int(duration * sampling_rate))
    pulse_train = np.zeros_like(time)

    # Generate the pulse train
    for i, t in enumerate(time):
        if (t % pulse_period) < pulse_duration:
            pulse_train[i] = pulse_amplitude

    return time, pulse_train


def generate_pulse_sequence(
    pulse_rate, pulse_amplitude, fill_factor, bit_stream, sampling_rate
):
    """
    Returns the pulse sequence based on the given bit stream.

    Parameters:
    ----------
    pulse_rate : float
        The rate at which pulses occur in the sequence, in pulses per second.

    pulse_amplitude : float
        The amplitude of each pulse in the sequence.

    fill_factor : float
        The ratio of the pulse duration to the pulse period (pulse duration / pulse period).

    bit_stream : np.array
        A binary data stream (array of 0s and 1s) used to modulate the pulse sequence.

    sampling_rate : float
        The sampling rate in samples per second, dictating the resolution of the pulse sequence.

    Returns:
    --------
    time : np.array
        An array representing the time intervals at which the pulse sequence is sampled.

    pulse_sequence : np.array
        The generated pulse sequence signal as an array, corresponding to the time array.

    """

    # Calculate pulse duration based on the filling factor
    pulse_period = 1 / pulse_rate
    pulse_duration = pulse_period * fill_factor

    # Total duration based on the pattern, seconds
    total_duration = len(bit_stream) * pulse_period

    # Create a time vector
    time = np.linspace(0, total_duration, int(total_duration * sampling_rate))
    pulse_sequence = np.zeros_like(time)

    # Generate the pulse sequence
    for i, t in enumerate(time):
        pattern_index = int(t // pulse_period)
        if pattern_index < len(bit_stream) and bit_stream[pattern_index]:
            if (t % pulse_period) < pulse_duration:
                pulse_sequence[i] = pulse_amplitude

    return time, pulse_sequence


def am(
    carrier_frequency, carrier_amplitude, amplitude_sensitivity, baseband_signal, time
):
    """
    Returns the amplitude modulated signal.

    Parameters:
    ----------
    carrier_frequency : float
        The frequency of the carrier wave in Hertz (Hz).

    carrier_amplitude : float
        The amplitude of the carrier wave.

    amplitude_sensitivity : float
        The sensitivity of the amplitude modulation, in V^(-1), assuming the modulating signal is in volts.

    baseband_signal : np.array
        The baseband (modulating) signal.

    time : np.array
        The time vector over which the modulation occurs.

    Returns:
    --------
    np.array
        The amplitude modulated signal as an array.
    """

    return (
        carrier_amplitude
        * (1 + amplitude_sensitivity * baseband_signal)
        * np.cos(2 * np.pi * carrier_frequency * time)
    )


def pm(carrier_frequency, carrier_amplitude, phase_sensitivity, baseband_signal, time):
    """
    Returns the phase modulated signal.

    Parameters:
    ----------
    carrier_frequency : float
        The frequency of the carrier signal in Hertz (Hz).

    carrier_amplitude : float
        The amplitude of the carrier signal.

    phase_sensitivity : float
        The sensitivity of the phase modulation, in radians per volt (rad/V), assuming the modulating signal is in volts.

    baseband_signal : np.array
        The baseband (modulating) signal.

    time : np.array
        The time vector over which the modulation occurs.

    Returns:
    --------
    np.array
        The phase modulated signal as an array.
    """

    return carrier_amplitude * np.cos(
        2 * np.pi * carrier_frequency * time + phase_sensitivity * baseband_signal
    )


def qam(carrier_frequency, constellation, bit_stream, symbol_rate, sampling_rate):
    """
    Returns the QAM modulated signal based on the given constellation and bit stream.

    This function takes a binary data stream and modulates it onto a carrier wave
    using the QAM technique. It generates a QAM signal by mapping binary tuples to
    corresponding I/Q amplitudes based on the provided constellation, and then
    modulating these I/Q components onto a sine and cosine carrier wave.

    Parameters:
    ------------
    carrier_frequency : float
        The frequency of the carrier signal in Hertz (Hz).

    constellation : dict
        A dictionary mapping binary tuples to I/Q amplitudes. Each key is a tuple
        representing a binary sequence, and its value is a tuple (I, Q) representing
        the in-phase and quadrature amplitudes.

    bit_stream : np.array
        An array of binary data (0s and 1s) to be modulated. The length of this array
        should be a multiple of the number of bits represented by each constellation point.

    symbol_rate : float
        The rate at which symbols are transmitted, in symbols per second. This is
        inversely related to the duration of each symbol.

    sampling_rate : float
        The rate at which the signal is sampled, in samples per second. This determines
        the number of data points used to represent the signal in one second.

    Returns:
    --------
    (time, qam_signal) : tuple
        A tuple (time, qam_signal), where 'time' is a numpy array representing the time
        coordinates for the signal, and 'qam_signal' is the modulated QAM signal as a
        numpy array.

    Examples:
    ---------
    # Example usage of the function:
    carrier_freq = 1000  # 1000 Hz
    constellation = {(0, 0): (-1, -1), (0, 1): (-1, 1), (1, 0): (1, -1), (1, 1): (1, 1)}
    bit_stream = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    symbol_rate = 500   # 500 symbols per second
    sampling_rate = 10000  # 10000 samples per second

    time, qam_signal = qam(carrier_freq, constellation, bit_stream, symbol_rate, sampling_rate)
    """

    if not (len(constellation) & (len(constellation) - 1) == 0):
        raise ValueError("Number of constellation points must be a power of 2.")

    bits_per_symbol = int(np.log2(len(constellation)))

    if len(bit_stream) % bits_per_symbol != 0:
        raise ValueError("Length of data must be a multiple of bits per symbol.")

    num_symbols = len(bit_stream) // bits_per_symbol
    symbol_duration = 1 / symbol_rate
    samples_per_symbol = int(sampling_rate / symbol_rate)

    # Creating time array for entire signal
    time = np.linspace(
        0,
        num_symbols * symbol_duration,
        num_symbols * samples_per_symbol,
        endpoint=False,
    )

    # Initialize I and Q components
    I_component = np.zeros_like(time)
    Q_component = np.zeros_like(time)

    # Efficiently assign I and Q values for each symbol
    for i in range(num_symbols):
        start_index = i * samples_per_symbol
        symbol_bits = tuple(bit_stream[i * bits_per_symbol : (i + 1) * bits_per_symbol])
        I, Q = constellation[symbol_bits]
        I_component[start_index : start_index + samples_per_symbol] = I
        Q_component[start_index : start_index + samples_per_symbol] = Q

    # Generate carrier signals
    carrier_I = np.cos(2 * np.pi * carrier_frequency * time)
    carrier_Q = np.sin(2 * np.pi * carrier_frequency * time)

    # Modulate the carriers
    modulated_I = I_component * carrier_I
    modulated_Q = Q_component * carrier_Q

    # Combine the modulated signals
    qam_signal = modulated_I + modulated_Q

    return time, qam_signal


def apply_noise(signal, noise_level):
    """
    Returns the signal with added Gaussian noise.

    Parameters:
    ----------
    signal : np.array
        The original signal to which noise is to be added.

    noise_level : float
        The standard deviation of the Gaussian noise. This value determines the noise intensity.

    Returns:
    --------
    np.array
        The signal with added Gaussian noise. The noise is added on a per-sample basis.
    """

    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise


def apply_attenuation(signal, attenuation_factor):
    """
    Returns the attenuated signal.

    Parameters:
    ----------
    signal : np.array
        The original signal to be attenuated.

    attenuation_factor : float
        The factor by which the signal is attenuated. A value less than 1 reduces the signal amplitude.

    Returns:
    --------
    np.array
        The attenuated signal. The signal's amplitude is scaled, but its length and shape remain unchanged.
    """

    return signal * attenuation_factor


def spectrum_analyzer(signal, sampling_rate):
    """
    Returns the frequency and magnitude spectrum of a signal.

    Parameters:
    ----------
    signal : np.array
        The input signal whose frequency spectrum is to be analyzed.

    sampling_rate : float
        The sampling rate of the signal in samples per second. This affects the frequency resolution.

    Returns:
    --------
    frequency : np.array
        An array of frequencies present in the signal, up to the Nyquist frequency.

    magnitude : np.array
        An array representing the magnitude (or amplitude) at each frequency in the frequency array.

    Notes:
    ------
    - The function returns only the positive half of the frequency spectrum, as it is symmetrical for real signals.
    - The magnitude spectrum is normalized by the number of samples in the signal.
    """

    n = len(signal)
    frequency = np.fft.fftfreq(n, d=1 / sampling_rate)
    magnitude = np.abs(np.fft.fft(signal)) / n
    return (
        frequency[: n // 2],
        magnitude[: n // 2],
    )
