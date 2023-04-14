declare -a dataset=("ogbg-moltoxcast" "ogbg-molsider" "ogbg-mollipo" "ogbg-molbace" "ogbg-molbbbp" "ogbg-moltox21" "ogbg-molesol" "ogbg-molclintox")
declare -a counts_cur_path=("OGBG-MOLTOXCAST_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLSIDER_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLLIPO_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLBACE_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLBBBP_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLTOX21_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLESOL_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLCLINTOX_full_kernel_max_50_run1.singleton_filtered" "OGBG-MOLHIV_full_kernel_max_50_run1.singleton_filtered")


# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"

    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/${counts_cur_path[i]}"    
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10

    python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --repeats 2 --nr_diff_counts 9
    done