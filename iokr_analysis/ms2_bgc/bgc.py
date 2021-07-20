from kernel_util import l2_gaussian
import csv
import numpy
import os


def wv_kernel(v1, v2):
    return l2_gaussian(numpy.array(v1), numpy.array(v2))


class Corpus(object):
    def __init__(self):
        self.dictionary = set([])
        self.documents = []

    def add_document(self, document):
        self.documents.append(document)
        self.dictionary = self.dictionary.union(document.words)

    def initialise(self):
        # build the dict and the word vectors
        self.dictionary = list(self.dictionary)
        self.dictionary.sort()
        print('building word vectors from {} words'.format(len(self.dictionary)))
        for d in self.documents:
            d_wv = [1 if x in d.words else 0 for x in self.dictionary]
            d.word_vector = d_wv

    def to_wv(self, word_list):
        return [1 if x in word_list else 0 for x in self.dictionary]


class BGC(object):
    def __init__(self, bgc_id):
        self.bgc_id = bgc_id
        self.pfam_domains = []
        self.metabolites = []
        self.word_vector = []

    @property
    def words(self):
        return self.pfam_domains


def load_data_dir(mibig_structure_file, pfs_data_dir):
    corpus = Corpus()

    bgc_structures = {}
    with open(mibig_structure_file, 'r') as f:
        r = csv.reader(f)
        header = next(r)
        for line in r:
            bgc_id, bgc_name, smiles, pubchem = line
            if smiles.strip() == "":
                continue

            if bgc_id in bgc_structures:
                bgc_structures[bgc_id].append(smiles)
            else:
                bgc_structures[bgc_id] = [smiles]

    for filename in os.listdir(pfs_data_dir):
        if not filename.endswith('.pfs'):
            continue
        bgc_id = filename.split('.')[0]
        if bgc_id in bgc_structures:
            with open(pfs_data_dir + os.sep + filename, 'r') as f:
                line = f.read()
                domains = line.split()

            bgc = BGC(bgc_id)
            bgc.pfam_domains = domains
            bgc.metabolites = bgc_structures[bgc_id]

            corpus.add_document(bgc)

    corpus.initialise()

    return corpus
