"""
After having already run an experiment with a specific count, this script redoes the evaluation with different counts
and the same hyperparameters.
"""

import argparse
import glob
import os


from Exp.run_experiment import main as run_exp

def main():
    parser = argparse.ArgumentParser(description='An experiment.')
    parser.add_argument('-grid', dest='grid_file', type=str,
                    help="Path to a .yaml file that contains the parameters grid.")
    parser.add_argument('-dataset', type=str)
    parser.add_argument('--repeats', type=int, default=10,
                    help="Number of times to repeat the final model training")
    parser.add_argument('--folds', type=int, default="1",
                    help='Number of folds, setting this to something other than 1, means we will treat this as cross validation')
    parser.add_argument('--nr_diff_counts', type=int, default="5",
                    help='Number of different graph feature files for which the evaluation will be ran')
    
    args = parser.parse_args()

    # Collect all relevant Counts
    relevant_new_counts = glob.glob(os.path.join(".", "Counts", "RepeatedCounts", f"{args.dataset.upper()}_*.overflow_filtered"))
    relevant_new_counts = list(filter(lambda x: int(x.split("run")[1].split(".")[0]) <= args.nr_diff_counts, relevant_new_counts))
    assert len(relevant_new_counts) == args.nr_diff_counts

    old_count = glob.glob(os.path.join(".", "Counts", "Current", f"{args.dataset.upper()}_*"))[0]
    old_count_filename = os.path.split(old_count)[-1]

    for new_count in relevant_new_counts:
        args = {
            "-grid": args.grid_file,
            "-dataset": args.dataset,
            "--repeats": args.repeats,
            "--folds": args.folds,
            "--graph_feat": new_count,
            "--params_exp_name":  f"{args.dataset}_{os.path.split(args.grid_file)[-1]}" + f"_{old_count_filename}" 
        }
        run_exp(args)


if __name__ == "__main__":
    main()