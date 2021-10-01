dataset_arr=( /home/compu/data/long/projects/roi_segmentation/roi_segmentation_dataset.csv
              /home/compu/data/long/projects/roi_segmentation/small_lesion_dataset.csv
              /home/compu/data/long/projects/roi_segmentation/medium_lesion_dataset.csv
              /home/compu/data/long/projects/roi_segmentation/large_lesion_datast.csv )
for u in "${dataset_arr[@]}"
do
  echo $u
  PYTHONPATH=. python criteria/dice_loss.py $u
done
