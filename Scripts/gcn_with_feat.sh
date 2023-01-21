declare -a dataset=("ogbg-moltoxcast" "ogbg-molsider" "ogbg-mollipo" "ogbg-molbace" "ogbg-molbbbp" "ogbg-moltox21" "ogbg-molesol" "ogbg-molclintox" "ogbg-molhiv")
declare -a counts_cur_path=("OGBG-MOLTOXCAST_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLSIDER_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLLIPO_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLBACE_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLBBBP_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLTOX21_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLESOL_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLCLINTOX_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLHIV_full_kernel_max_50_il3.overflow_filtered")

# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"
    echo "Counts/Current/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10 --graph_feat "Counts/Current/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    done

# ZINC
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_with_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10 --graph_feat "Counts/Current/ZINC_SUBSET_full_kernel_max_50_20230117.overflow_filtered"
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_with_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10