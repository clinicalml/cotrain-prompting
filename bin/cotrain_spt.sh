export TOKENIZERS_PARALLELISM=false

for seed in 0 1 32 42 1024
do
    for dataset in rte cb
    do
        python -m src.cotrain -c t03b.json+${dataset}.json+prompt_tuning-10_prompts.json -k exp_name=cotrain_spt_${dataset}_seed${seed} seed=${seed} few_shot=False allow_skip_exp=True train_template_idx=0 eval_template_idx=0 eval_epoch_interval=1 prompt_tuning_num_prefix_emb=20 prompt_tuning_decoder=False num_steps=30000 prompt_tuning_init_with_pad=True cotrain_load_best=True batch_size=16 grad_accum_factor=2 > cotrain_spt_newval_${dataset}_seed${seed}.log 2>&1
    done
done
