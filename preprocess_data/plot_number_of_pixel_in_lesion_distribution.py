import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    number_of_pixels_in_lesion = []
    first_line = True
    with open("/home/longlh/hard_2/PycharmProjects/roi_segmentation/testing_data_lesion_metadata.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if first_line:
                first_line = False
                continue
            number_of_pixels_in_lesion.append(int(row[3]))

    # number pixels of large lesion >= 500,
    # 50 < number pixels of medium lesion < 500,
    # number pixels of small lesion <= 50
    large_lesion = [i for i in number_of_pixels_in_lesion if i >= 500]
    medium_lesion = [i for i in number_of_pixels_in_lesion if 50 < i < 500]
    small_lesion = [i for i in number_of_pixels_in_lesion if i <= 50]
    super_small_lesion = [i for i in number_of_pixels_in_lesion if i <= 10]
    print(len(small_lesion))
    print(len(medium_lesion))
    print(len(large_lesion))
    plt.hist(number_of_pixels_in_lesion)
    # plt.hist(small_lesion)
    # plt.hist(medium_lesion)
    # plt.hist(large_lesion)
    plt.show()