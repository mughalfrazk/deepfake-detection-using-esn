import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from models.FaceLandmarkModule import FaceLandmarkGenerator
import glob
import seaborn as sns
import os

def pad_to_max_length(array, max_length, pad_value = 0):
    if array.ndim == 2:
        padded = np.pad(array, ((0, max_length - len(array)), (0, 0)), mode="constant", constant_values=pad_value)
    else:
        padded = np.pad(array, (0, max_length - len(array)), mode='constant', constant_values=pad_value)

    return padded


if __name__ == "__main__":

    fake_dir = "/Users/mughalfrazk/Study/SHU/Dissertation/code/mediapipe-eye-detection/out/0"
    orig_dir = "/Users/mughalfrazk/Study/SHU/Dissertation/code/mediapipe-eye-detection/out/1"
    
    fake_paths = glob.glob(fake_dir + "/*.npy")
    orig_paths = glob.glob(orig_dir + "/*.npy")
    
    all_features = []
    all_targets = []

    loaded_fakes = []
    loaded_origs = []

    max_frames = 0
    for path in fake_paths:
        target_label = 0
        loaded = np.load(path)
        if loaded.ndim == 2:
            loaded_fakes.append(loaded)
            frames, features = loaded.shape
        
            print("Frames updated: ", frames)
            if frames > max_frames:
                max_frames = frames
                print("Max frames updated: ", max_frames)

    for path in orig_paths:
        target_label = 1
        loaded = np.load(path)
        if loaded.ndim == 2:
            loaded_origs.append(loaded)
            frames, features = loaded.shape
        
            print("Frames updated: ", frames)
            if frames > max_frames:
                max_frames = frames
                print("Max frames updated: ", max_frames)


    fake_target = np.array([0] * max_frames)
    for np_arr in loaded_fakes:

        padded_arr = pad_to_max_length(np_arr, max_length=max_frames)
        all_features.append(padded_arr)
        all_targets.append(fake_target)


    orig_target = np.array([1] * max_frames)
    for np_arr in loaded_fakes:

        padded_arr = pad_to_max_length(np_arr, max_length=max_frames)
        all_features.append(padded_arr)
        all_targets.append(orig_target)


    # Convert lists to 3D and 2D NumPy arrays
    X_dataset = np.stack(all_features)  # Shape: (videos, frames, features)
    Y_dataset = np.stack(all_targets)   # Shape: (videos, frames)
      
    print("X_dataset: ", X_dataset.shape)
    print("Y_dataset: ", Y_dataset.shape)


    # if not(os.path.exists(f"./out/ff_dataset_80")):
    #     os.makedirs("./out", exist_ok=True)
    #     ds = {"ORE_MAX_GIORNATA": 5}
    #     np.savez_compressed(os.path.join("./out/", "ff_dataset_80"), ds)

    np.savez_compressed("./out/ff_dataset_80", X_dataset=X_dataset, Y_dataset=Y_dataset)
    