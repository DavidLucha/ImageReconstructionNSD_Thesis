## Expanded_AFNI_Process
Custom AFNI script to get all the labels from each subject's ROIs. Also includes code for resampling ROIs to 3mm using AFNI's 3dresample
Combines the ROIs into a full mask and recodes the voxels to quickly determine which are overlapping.

Codes for these are as follows:

- 1 V1v	    
- 2 V1d	   
- 3 V2v	   
- 4 V2d	   
- 5 V3v	    
- 6 V3d	    
- 7 hV4
- 40 OFA
- 50 FFA-1 
- 60 FFA-2 
- 70 mTL-faces 
- 80 aTL-faces
- 200 OPA  
- 300 PPA 
- 400 RSC
- 1000 EBA  
- 2000 FBA-1  
- 3000 FBA-2  
- 4000 mTL-bodies 

The breakdowns for this can be found in /NSD_AFNI_data/roi_lists/ROI Lists V2.xlsx

## Final_AFNI_Per_Subject_fixRAND
Custom AFNI script which loads each session from the NSD, converts these to float and divides by 300 (as per NSD data manual). 
Then, this samples from random voxels outside of non-visual areas.

## Final_AFNI_Per_Subject
Custom AFNI script which resamples each ROI to 3mm and makes combined masks for both resolutions
Loads each session from the NSD, converts these to float and divides by 300 (as per NSD data manual). 
Extracts all voxel data from each ROI at each voxel resolution.
Note: the random sample code here needs to be replaced with that from Final_AFNI_Per_Subject_fixRAND.

## Whole_Brain_Mask_NEW
Custom AFNI script to define all visual areas manually defined using pRF and fLoc experiments in NSD. We use this to subtract from the whole brain data, so we can randomly sample from 'non-visual areas'.


### Notes:
- There's some general double handling in these scripts as I had some issues I had to fix post-hoc.
- Final masks can be found at /NSD_AFNI_data/

