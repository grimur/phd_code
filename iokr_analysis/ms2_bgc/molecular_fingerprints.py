from kernel_util import l2_gaussian, gaussian
import scipy.io
from cdk_pywrapper.cdk_pywrapper import Compound
import pickle
import numpy


# def fingerprint_kernel(fp1, fp2, gamma=0.01):
#     # vectorised kernel function
#     d_sq = numpy.sum(numpy.power((fp1 - fp2), 2), axis=1)
#    return numpy.exp(- gamma * d_sq)


def fingerprint_kernel(v1, v2):
    v1 = v1.astype('bool')
    v2 = v2.astype('bool')
    tanimoto_or = numpy.count_nonzero(numpy.logical_or(v1, v2))
    tanimoto_and = numpy.count_nonzero(numpy.logical_and(v1, v2))
    if tanimoto_and is 0:
        tanimoto = 0
    else:
        tanimoto = tanimoto_or / tanimoto_and
    return gaussian(tanimoto ** 2)


def main(datapath):
    data = scipy.io.loadmat(datapath + '/data_GNPS.mat')

    fp_size = 1024

    new_fps = []

    for i in data['inchi']:
        print('Processing sample %s' % i)
        inchi = i[0][0]
        print(inchi)
        c = Compound(compound_string=inchi, identifier_type='inchi')
        fp = c.get_fingerprint()
        fp_array = numpy.zeros(fp_size)
        for fp_bit in range(fp_size):
            fp_array[fp_bit] = fp.get(fp_bit)
        new_fps.append(fp_array)

    with open(datapath + '/cdk_fingerprints.bin', 'wb') as f:
        pickle.dump(new_fps, f)


def default_fingerprint_from_inchi(inchi):
    c = Compound(compound_string=inchi, identifier_type='inchi')
    # this should accommodate different types!
    fp = c.get_fingerprint()
    fp_size = 1024
    fp_array = numpy.zeros(fp_size)
    for fp_bit in range(fp_size):
        fp_array[fp_bit] = fp.get(fp_bit)
    return fp_array


def fingerprint_from_smiles(smiles, fingerprint_type=None):
    if fingerprint_type is None:
        fingerprint = numpy.array([])
        for fp_type in ('cdk_default', 'substructure', 'klekota-roth'):
            fingerprint = numpy.hstack((fingerprint, fingerprint_from_smiles(smiles, fp_type)))
        return fingerprint

    c = Compound(compound_string=smiles, identifier_type='smiles')

    if fingerprint_type == 'cdk_default':
        fingerprinter = c.cdk.fingerprint.Fingerprinter()
    elif fingerprint_type == 'substructure':
        fingerprinter = c.cdk.fingerprint.SubstructureFingerprinter()
    elif fingerprint_type == 'klekota-roth':
        fingerprinter = c.cdk.fingerprint.KlekotaRothFingerprinter()
    else:
        raise SystemExit('Unknown fingerprint type: {}'.format(fingerprint_type))

    fp = fingerprinter.getBitFingerprint(c.mol_container)
    fp_size = fp.size()
    fp_array = numpy.zeros(fp_size)
    for fp_bit in range(fp_size):
        fp_array[fp_bit] = fp.get(fp_bit)
    return fp_array


def fingerprint_from_inchi(inchi, fingerprint_type=None):
    if fingerprint_type is None:
        fingerprint = numpy.array([])
        for fp_type in ('cdk_default', 'substructure', 'klekota-roth'):
            fingerprint = numpy.hstack((fingerprint, fingerprint_from_inchi(inchi, fp_type)))
        return fingerprint

    c = Compound(compound_string=inchi, identifier_type='inchi')

    if fingerprint_type == 'cdk_default':
        fingerprinter = c.cdk.fingerprint.Fingerprinter()
    elif fingerprint_type == 'substructure':
        fingerprinter = c.cdk.fingerprint.SubstructureFingerprinter()
    elif fingerprint_type == 'klekota-roth':
        fingerprinter = c.cdk.fingerprint.KlekotaRothFingerprinter()
    else:
        raise SystemExit('Unknown fingerprint type: {}'.format(fingerprint_type))

    fp = fingerprinter.getBitFingerprint(c.mol_container)
    fp_size = fp.size()
    fp_array = numpy.zeros(fp_size)
    for fp_bit in range(fp_size):
        fp_array[fp_bit] = fp.get(fp_bit)
    return fp_array


if __name__ == '__main__':
    main()
