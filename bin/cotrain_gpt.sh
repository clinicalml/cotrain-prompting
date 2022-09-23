export TOKENIZERS_PARALLELISM=false

for seed in 0 1 32 42 1024
do
    for dataset in rte cb trec
    do
        python -m src.cotrain_gpt -c gpt-${dataset}.json -k exp_name=cotrain_gpt_${dataset}_seed${seed} seed=${seed} few_shot=False allow_skip_exp=True eval_epoch_interval=1 cotrain_load_best=True num_epochs=40 lr=0.1 weight_decay=5e-3 cotrain_min_pct_per_class=-1 > cotrain_gpt_${dataset}_seed${seed}.log 2>&1
    done
done
