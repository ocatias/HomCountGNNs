declare -a dataset=("ogbg-molbace" "ogbg-molbbbp" "ogbg-moltox21" "ogbg-molesol" "ogbg-molclintox")
# declare -a dataset=("ogbg-molbace" "ogbg-molbbbp" "ogbg-moltox21" "ogbg-molesol" "ogbg-molclintox" "ogbg-molsider"  "ogbg-moltoxcast" "ogbg-mollipo")
declare -a counts_cur_path=("OGBG-MOLBACE_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLBBBP_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLTOX21_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLESOL_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLCLINTOX_full_kernel_max_50_il3.overflow_filtered")
# declare -a counts_baseline_path=("OGBG-MOLBACE_tree+cycle_6_50_il3.homson" "OGBG-MOLBBBP_tree+cycle_6_50_il3.homson")



# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"
    echo "Counts/Current/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/with_features.yaml" -dataset "${dataset[i]}" --candidates 40  --repeats 10 --graph_feat "Counts/Current/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/with_features.yaml" -dataset "${dataset[i]}" --candidates 40  --repeats 10
    # python Exp/run_experiment.py -grid "Configs/Eval/with_features.yaml" -dataset "${dataset[i]}" --candidates 40  --repeats 10 --graph_feat "Counts/Baselines/${counts_baseline_path[i]}"
    done


