#### glove+TraceNEt
 CUDA_VISIBLE_DEVICES='1'  python run_glove_input.py  --output_hidden_states   --output_item_weights  --per_gpu_eval_batch_size 64   --per_gpu_train_batch_size 64     --max_seq_length 256  --learning_rate 0.001   --output_feature 50   --dropout_prob 0.2  --num_train_epochs  10  --task yelp-5  --weight_decay   0.2   --proactive_masking  --seq_select_prob 0.05  --seed 1
 
 ### TraceNet-G
> CUDA_VISIBLE_DEVICES='3'  python run_glove_attn.py  --per_gpu_eval_batch_size 64   --per_gpu_train_batch_size 64     --max_seq_length 256  --learning_rate 0.001   --output_feature 50   --dropout_prob 0.2  --num_train_epochs  10  --task yelp-5  --weight_decay   0.2  --num_hubo_layers 3  --seed 1

CUDA_VISIBLE_DEVICES='3'  python run_glove_attn.py  --per_gpu_eval_batch_size 64   --per_gpu_train_batch_size 64     --max_seq_length 256  --learning_rate 0.001   --output_feature 50   --dropout_prob 0.2  --num_train_epochs  10  --task yelp-5  --weight_decay   0.2  --num_hubo_layers 3  --seed 1