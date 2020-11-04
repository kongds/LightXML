#!/bin/bash

if [ "$1" = "amazon670k" ]; then
    echo start $1
    for i in 0 1 2
    do
        python ./src/cluster.py --dataset amazon670k --id $1
	    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i
	    python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i --eval_model
    done
    python src/ensemble_direct.py --model1 amazon670k-t0 --model2 amazon670k-t1 --model3 amazon670k-t3 --dataset amazon670k
elif [ "$1" = "wiki500k" ]; then
    echo start $1
    for i in 0 1 2
    do
        python ./src/cluster.py --dataset wiki500k --id $1
	    python src/main.py --lr 1e-4 --epoch 10 --dataset wiki500k --swa --swa_warmup 4 --swa_step 3000  --batch 32 --max_len 128 --eval_step 3000  --group_y_candidate_num 2000 --group_y_candidate_topk 32 --valid --hidden_dim 500 --group_y_group $i
	    python src/main.py --lr 1e-4 --epoch 10 --dataset wiki500k --swa --swa_warmup 4 --swa_step 3000  --batch 32 --max_len 128 --eval_step 3000  --group_y_candidate_num 2000 --group_y_candidate_topk 32 --valid --hidden_dim 500 --group_y_group $i --eval_model
    done
    python src/ensemble_direct.py --model1 wiki500k-t0 --model2 wiki500k-t1 --model3 wiki500k-t3 --dataset wiki500k
elif [ "$1" = "amazoncat13k" ]; then
    echo start $1
    #train
    python src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000
    python src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000 --bert roberta
    python src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 32 --eval_step 20000 --bert xlnet --max_len 128

    python src/ensemble.py --dataset amazoncat13k
elif [ "$1" = "wiki31k" ]; then
    echo start $1
    python src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 300 --batch 16
    python src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2  --bert xlnet
    python src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2  --bert roberta

    python src/ensemble.py --dataset wiki31k
elif [ "$1" = "eurlex4k" ]; then
    echo start $1
    python src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 200 --batch 16  
    python src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 200 --batch 16  --bert roberta
    python src/main.py --lr 5e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet

    python src/ensemble.py --dataset eurlex4k
fi
