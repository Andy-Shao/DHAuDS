#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AST.CochlScene.ttda --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --cache_path $BASE_PATH'/tmp' \
    --max_epoch 120 --lr_cardinality 50 --batch_size 15 --lr '5.5e-5' --num_workers 16 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 35 \
    --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --corruption_type 'WHNP' --corruption_level 'L2' \
    --ast_wght_pth './result/CochlScene/AST/Train/AST-CS-ast.pt' \
    --clsf_wght_pth './result/CochlScene/AST/Train/AST-CS-clsf.pt'