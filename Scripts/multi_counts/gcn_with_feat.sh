declare -a dataset=("ogbg-moltoxcast" "ogbg-molsider" "ogbg-mollipo" "ogbg-molbace" "ogbg-molbbbp" "ogbg-moltox21" "ogbg-molesol" "ogbg-molclintox" "ogbg-molhiv")

# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"
    python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --repeats 10 --nr_diff_counts 5
    done

# ZINC
python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "ZINC" --repeats 10 --nr_diff_counts 5