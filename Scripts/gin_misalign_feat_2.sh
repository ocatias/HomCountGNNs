declare -a dataset=("ogbg-molbbbp" "ogbg-moltox21")
declare -a counts_cur_path=("OGBG-MOLBBBP_full_kernel_max_50_il3.overflow_filtered" "OGBG-MOLTOX21_full_kernel_max_50_il3.overflow_filtered")

# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"
    echo "Counts/Current/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gin_misaligned_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10 --graph_feat "Counts/Current/${counts_cur_path[i]}"
    done
