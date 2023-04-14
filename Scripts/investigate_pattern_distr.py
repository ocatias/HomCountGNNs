import glob
import os
import numpy as np
import json

import Misc.config as config

def main():
    
    file_paths = glob.glob(os.path.join(config.COUNTS_PATH, "*.singleton_filtered"))
    filenames = list(map(lambda f: os.path.split(f)[-1], file_paths))
    datasets = list(set(map(lambda f: f.split('_')[0], filenames)))
    
    for dataset in datasets:
        print(dataset)

        for file_path in filter(lambda f: dataset in f, file_paths):
            print(f"\t{file_path}")
            
            with open(file_path) as file:
                data = json.load(file)
                
                training_counts = list(map(lambda d: d["counts"], filter(lambda d: d["split"] == "train", data["data"])))
                avg = np.average(np.array(training_counts), axis = 0)
                
                print(np.array(training_counts).shape)
                # print(training_counts)
                print(f"avg: {avg}")
                
                quit()

if __name__ == "__main__":
    main()