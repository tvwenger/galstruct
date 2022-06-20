@echo off
@echo Starting Comparisons
python compare_models.py data/varysims_maf data/varysims_maf --variedparam nsims
python compare_models.py data/varysims_nsf data/varysims_nsf --variedparam nsims 
python compare_models.py data/varyfeatures_maf data/varyfeatures_maf --variedparam feat  
python compare_models.py data/varyfeatures_nsf data/varyfeatures_nsf --variedparam feat  
python compare_models.py data/varylayers_maf data/varylayers_maf --variedparam layers  
python compare_models.py data/varylayers_nsf data/varylayers_nsf --variedparam layers 
