
## Data Organization

```
partitions/
    |---curated_test/ (testing subsets definition for each ICAO requirement)
        |---blurred/
            |---cmp_df.csv
            |---uncmp_df.csv
        |---dark_tinted_lenses/
            |---cmp_df.csv
            |---uncmp_df.csv
        |...
    |---ids_all_train.txt (list with all the trainings subject ids)
    |---ids_balanced_test.txt (list with all the testing subject ids)
    |---ids_balanced_train.txt (subset of training subject ids balanced accross demographic groups)

preprocessed/ (face+torso cropped data)
    |---Artificial_Torso/ (artificially generated images)
        |---subjectID/
            |---Camera/
                |---img_name.JPG
                |...
        |...

    |---Data_Torso/ (original cropped images)
        |---subjectID/
            |---Camera/
                |---img_name.JPG
                |...
            |---Mobile/
                |---img_name.JPG
                |...
        |...

    |---LandMarks_Torso/ (facial landmarks for images in Data_Torso/)
        |---subjectID/
            |---Camera/
                |---img_name_spiga.pickle
                |...
            |---Mobile/
                |---img_name_spiga.pickle
                |...
        |...

    |---Masks_Torso/ (segmentation masks for images in Data_Torso/)
        |---subjectID/
            |---Camera/
                |---img_name_masks.pickle
                |...
        |...


raw/ (uncropped, raw media for each subject: HQ, SQ and videos)
    |---subjectID/
        |---Camera/
            |---img_name.JPG
            |...
        |---Mobile/
            |---img_name.JPG
            |...
        |---Videos/
            |---vid_name.MP4
            |...
    |...

artificial_image_labels.csv (table with ICAO requirement labelling definition for all generated images in **Artificial_Torso/** directory)

camera_image_labels.csv (table with ICAO requirement labelling definition for all original HQ images in **Data_Torso/../Camera/** directory)

demographic_info.csv (demographic informations for each subjectID)
```
