
import numpy
import spectrum


def ppk_wrapper(ms_i, ms_j):
    sigma_mass = 0.00001
    sigma_int = 1000000.0
    ppk_peaks = spectrum.ppk(ms_i.spectrum, ms_j.spectrum, sigma_mass, sigma_int)
    ppk_nloss = spectrum.ppk_nloss(ms_i.spectrum, ms_j.spectrum, ms_i.parentmass, ms_j.parentmass, sigma_mass, sigma_int)

    if not hasattr(ms_i, 'ppk_peaks'):
        ms_i.ppk_peaks = spectrum.ppk(ms_i.spectrum, ms_i.spectrum, sigma_mass, sigma_int)
        ms_i.ppk_nloss = spectrum.ppk_nloss(ms_i.spectrum, ms_i.spectrum, ms_i.parentmass, ms_i.parentmass, sigma_mass, sigma_int)

    if not hasattr(ms_j, 'ppk_peaks'):
        ms_j.ppk_peaks = spectrum.ppk(ms_j.spectrum, ms_j.spectrum, sigma_mass, sigma_int)
        ms_j.ppk_nloss = spectrum.ppk_nloss(ms_j.spectrum, ms_j.spectrum, ms_j.parentmass, ms_j.parentmass, sigma_mass, sigma_int)

    ppk_peaks_normalised = ppk_peaks / numpy.sqrt(ms_i.ppk_peaks * ms_j.ppk_peaks)
    ppk_nloss_normalised = ppk_nloss / numpy.sqrt(ms_i.ppk_nloss * ms_j.ppk_nloss)
    ppk = (ppk_peaks_normalised + ppk_nloss_normalised) / 2
    return ppk

