export TOKENIZERS_PARALLELISM=false

for seed in 0 1 32 42 1024
do
    for dataset in rte cb boolq
    do
        python -m src.cotrain -c t03b.json+${dataset}.json+ia3.json -k exp_name=cotrain_ia3_${dataset}_seed${seed} seed=${seed} few_shot=False allow_skip_exp=True train_template_idx=0 eval_template_idx=0 eval_epoch_interval=1 > cotrain_ia3_${dataset}_seed${seed}.log 2>&1
    done
done
