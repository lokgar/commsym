import numpy as np


def eopm(E_in, V_pi, V_drive):
    """
    Electro-optic phase modulator.

    Parameters:
    -----------
    E_in : scalar or np.array
        Input electric field of the optical carrier.

    V_pi : float
        Voltage required to achieve a phase shift of pi.

    V_drive : np.array
        Driving voltage signal (baseband/RF/etc.).

    Returns:
    --------
    E_out : np.array
        Output electric field of the optical carrier.
    """

    return E_in * np.exp(1j * np.pi * V_drive / V_pi)


def mzm(E_in, V_pi, V_bias, V_drive):
    """
    Mach-Zehnder modulator in push-pull configuration.

    Parameters:
    -----------
    E_in : scalar or np.array
        Input electric field of the optical carrier.

    V_pi : float
        Voltage required to achieve a phase shift of pi.

    V_bias : float
        Bias voltage applied to the modulator.

    V_drive : np.array
        Driving voltage signal (baseband/RF/etc.).

    Returns:
    --------
    E_out : np.array
        Output electric field of the optical carrier.
    """

    return E_in * np.cos(np.pi * (V_drive + V_bias) / V_pi)


def iqm(E_in, format, V_i, V_q):
    """
    IQ modulator.

    Parameters:
    -----------
    E_in : scalar or np.array
        Input electric field of the optical carrier.

    format : str
        Modulation format. Can be 'QPSK', '8QAM', '16QAM', '32QAM', '64QAM' or
        '128QAM'.

    V_i : np.array
        Driving voltage signal for the in-phase component.

    V_q : np.array
        Driving voltage signal for the quadrature component.

    Returns:
    --------
    E_out : np.array
        Output electric field of the optical carrier.
    """

    pass
