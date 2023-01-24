# ZINC hogsmeade
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10 --graph_feat "Counts/Current/ZINC_SUBSET_full_kernel_max_50_20230117.overflow_filtered"
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10

# ZINC slytherin
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10 --graph_feat "Counts/Current/ZINC_SUBSET_full_kernel_max_50_20230117.overflow_filtered"
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10
