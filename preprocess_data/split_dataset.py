import os

def split_dataset(root_dir, output_file, train_prop=0.6, val_prop=0.2):
    folders = os.listdir(root_dir)
    valid_folders = []
    for folder in folders:
        s_folders = os.listdir(f"{root_dir}/{folder}")
        if "ADC" in s_folders:
            s_s_folders = os.listdir(f"{root_dir}/{folder}/ADC")
            if "VOI2.xml" in s_s_folders:
                valid_folders.append(f"{root_dir}/{folder}/ADC")
    train_normal_num = int(len(valid_folders) * train_prop)
    val_normal_num = int(len(valid_folders) * val_prop)
    train_dataset = valid_folders[0:train_normal_num]
    val_dataset = valid_folders[train_normal_num: train_normal_num + val_normal_num]
    test_dataset = valid_folders[train_normal_num + val_normal_num:]
    # 0: train, 1: val, 2: test
    with open(output_file, 'w') as file:
        for a in train_dataset:
            file.writelines(f"{a},0\n")
        for b in val_dataset:
            file.writelines(f"{b},1\n")
        for c in test_dataset:
            file.writelines(f"{c},2\n")

if __name__ == '__main__':
    split_dataset("/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training",
                  "/home/longlh/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv")
