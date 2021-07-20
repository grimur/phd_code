import molecular_fingerprints
from cdk_pywrapper.cdk_pywrapper import Compound


class Metabolite(object):
    def __init__(self):
        self.structure = structure

        if structure.startswith('InChI'):
            self.fingerprint = molecular_fingerprints.fingerprint_from_inchi(structure)
        else:
            self.fingerprint = molecular_fingerprints.fingerprint_from_smiles(structure)


def compound(string):
    if string.startswith('InChI'):
        metabolite = Compound(compound_string=string, identifier_type='inchi')
    else:
        metabolite = Compound(compound_string=string, identifier_type='smiles')
    return metabolite


class MetaboliteServer(object):
    def __init__(self):
        # inchikey to fp
        self.metabolites = {}
        # structure to inchikey
        self.metabolite_lookup = {}
        pass

    def add_metabolite(self, structure):
        metabolite = compound(structure)

        inchikey = metabolite.get_inchi_key()

        if inchikey not in self.metabolites:
            inchi = metabolite.get_inchi()
            fingerprint = molecular_fingerprints.fingerprint_from_inchi(inchi)

            self.metabolites[inchikey] = {
                    'inchi': inchi,
                    'fingerprint': fingerprint
                    }

        if structure not in self.metabolite_lookup:
            self.metabolite_lookup[structure] = inchikey

    def get_metabolite(self, structure):
        if structure not in self.metabolite_lookup:
            metabolite = compound(structure)
            inchikey = metabolite.get_inchi_key()
            self.metabolite_lookup[structure] = inchikey
            if inchikey not in self.metabolites:
                inchi = metabolite.get_inchi()
                fingerprint = molecular_fingerprints.fingerprint_from_inchi(inchi)
                self.metabolites[inchikey] = {
                    'inchi': inchi,
                    'fingerprint': fingerprint
                }

        return self.metabolites[self.metabolite_lookup[structure]]
