"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
from   utils.utils     import walkdir
from   utils.detection import get_vehicle_coordinates
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # We iterate throught the data_folder (car_ims_v1):
    
    for file_path, file_name in walkdir(data_folder):

    #   2. Load the image
        img = cv2.imread(os.path.join(file_path, file_name))

    #   3. We get the coordinates of the box
        x1, y1, x2, y2 = get_vehicle_coordinates(img)

    #   4. We calculate aur region of interest:
        roi = img[ y1:y2 , x1:x2 ]

    #   5. We generate a new path in order to save the roi:

    #   car_ims_v2/train/Acura RL sedan/0000.90.jpg
    #   car_ims_v2     ---> output_data_folder
    #   train ot test  ---> sub_class_name = os.path.basename(os.path.dirname (file_path))
    #   Acura RL sedan ---> claa_name      = os.path.basename(file_path)

        class_name     = os.path.basename(file_path)
        sub_class_name = os.path.basename(os.path.dirname (file_path)) 
        
        new_path       = os.path.join(output_data_folder, sub_class_name, class_name)

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if not os.path.exists(os.path.join(new_path, file_name)):
            cv2.imwrite(os.path.join(new_path, file_name), roi)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
