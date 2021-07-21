# Ranking Microbial Metabolomic and Genomic Links using Complementary Scoring Functions

This repository contains the code used to run the experiments described in the PhD thesis *Ranking Microbial Metabolomic and Genomic Links using Compelmentary Scoring Functions*. 

The requirements.txt file describes the environment used to run the Jupyter notebooks. It surely includes packages that are not required, and does not include libraries that are. In particular, the NPLinker library (https://github.com/sdrogers/nplinker) and some other tools from Simon are used.

Other libraries may require manual install.

### data

The data folder contains the structural annotations for the MIBiG database, as well as a mock .mgf file containing the structure entries from GNPS without any ions. This may very well break some parsers.

The *microbial_data_sets* subfolder contains the config files needed for the NPLinker downloader to download the microbial data sets from PoDP.

### iokr_analysis

The *iokr_analysis* folder contains the scripts used to test the IOKR model. The MS2-MIBiG and BGC-MIBiG IOKR models uses the implementation from NPLinker. The MS2-BGC IOKR model is included in the directory.

### mibig_gnps_data_set

The notebooks required to assemble the MIBiG-GNPS data set

### phylogeny_adjustment

The notebooks used for the experiments on the phylogenetically adjusted strain correlation score

### strain_correlation

The notebook used to generate the graph of the expected value and variance of the strain correlation score. The code to test the strain correlation score on the microbial data set is contained in the iokr_analysis notebooks, along with the code for the MS2-MIBiG IOKR model.

### mibig_bgc_fp_distance_test.ipynb

Calculate the correlation between the BGC distance and the difference in molecular fingerprints for the associated metabolites.
