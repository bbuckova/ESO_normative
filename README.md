**Initial attempt with one visit and GPR**
----------------------

- freesurfer preprocessing done on sup servers (all results stored there)
>`fs_extract_stats.py`
- scrapes the necessary information - DK atlas and subcortical volumes and add to textfiles

>`clinics_load_desc.ipynb`
- initial descriptions of the dataset in form of jupyter notebook
- needs `clinics_desc)functions.py` as dependency

>`GPR_modelling.ipynb`
- contains code that evaluates GPR on the first visit of DK atlas 
- needs `clinics_desc)functions.py` as dependency

-------------------------
**Longitudinal analysis**
-------------------------
-------------------------
**Attempt 1**
- `fs_extract_stats_external_model.py`
    - this only extracts average thicknesses from a2009s + aseg