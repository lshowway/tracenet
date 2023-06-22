### README
This repository is the official implementation of TraceNet
#### Requirements
To install requirements:
> pip install -r requirements.txt
#### Dataset and Rosources
* [SST-5 dataset](https://nlp.stanford.edu/sentiment/) , or from [here](https://github.com/prrao87/fine-grained-sentiment) to down sentence-level SST-5 dataset
* Download [Yelp-5 dataset](http://goo.gl/JyCnZq), and then sample data by sample_yelp_data.py file.
*  [glove.840B.300d](https://nlp.stanford.edu/projects/glove/) is needed to generate  sst5.hdf5 file and yelp5.hdf5 file. Since glove.840B.300d is about 5GB, we recommend you use SST5.glove.to.TraceNet under dataset/SST_5, which is extracted from the former according to SST-5 vocabulary. 
* [SentiWordNet_3.0.0.txt](https://github.com/aesuli/SentiWordNet) is need for evaluation on attacks.
* The source code of CNN, LSTM, BiLSTM we modified is from [here](https://github.com/andyweizhao/capsule_text_classification) 
* The source code of Gumble Tree LSTM we modified is from [here](https://github.com/jihunchoi/unsupervised-treelstm)
#### Training
To train the model(s) in the paper, run this command (SST-5 dataset for example). Note that, each command and coresponded results are included in log.sst5 file:
> run CNN (rand, static, nonstatic, multichannel), LSTM, BiLSTM, under CNN_LSTM dir:
> first generate sst5.hdf5: `python sst_process.py`
>  and then: `python main.py  --dataset ../dataset/SST_5/sst5.hdf5  --model_type KIMCNN  --embedding_type rand  --num_classes 5  --batch_size 50  --max_sent 49   --num_epochs 20  --n_hidden 300`

-----------
>  run GT-LSTM
> rand : `python train.py   --task sst-5  --word-dim 300     --hidden-dim 300  --clf-hidden-dim 1024    --clf-num-layers 1  --dropout 0.5   --batch-size 64   --max_seq_length 49  --max-epoch 20  --pretrained False  --save-dir save/   --device cuda`

----
> run Bert, XLNet, Roberta
> bert: `python run_sentiment_classifier.py   --do_train   --do_eval   --output_dir  sst_bert_5307/  --model_type   bert    --model_name_or_path   ../bert_base_en  --per_gpu_eval_batch_size 1000   --task_name sst-5   --data_dir  ../dataset/SST_5/     --num_train_epochs 10.0   --dropout_prob 0.0   --output_feature 0   --num_hubo_layers  3  --method  null  --seq_select_prob 0.0  --per_gpu_train_batch_size 32   --max_seq_length 128   --learning_rate 5e-5    --seed  1 `
 
> xlnet: `python run_sentiment_classifier.py   --do_train   --do_eval   --output_dir  sst_xlnet_5533/    --overwrite_output_dir  --model_type   xlnet    --model_name_or_path   ../xlnet_base_cased  --per_gpu_eval_batch_size 1000   --task_name sst-5   --data_dir  ../dataset/SST_5/     --num_train_epochs 10.0   --dropout_prob 0.0   --output_feature 0   --num_hubo_layers  3  --method  null  --seq_select_prob 0.0  --per_gpu_train_batch_size 16   --max_seq_length 64   --learning_rate 2e-5    --seed  1 `

> roberta: `python run_sentiment_classifier.py   --do_train   --do_eval   --output_dir  sst_roberta/    --model_type roberta    --model_name_or_path roberta-base   --do_lower_case  --per_gpu_eval_batch_size 512   --task_name sst-5   --data_dir  ../dataset/sst5/     --num_train_epochs 10.0   --logging_steps  300  --dropout_prob 0.0   --output_feature 0   --num_hubo_layers  3  --method  null  --seq_select_prob 0.0  --per_gpu_train_batch_size 16   --max_seq_length 128   --learning_rate 2e-5    --seed  42 `
 
> roberta-yelp: `python run_sentiment_classifier.py   --do_train   --do_eval   --output_dir  yelp_roberta/    --model_type roberta    --model_name_or_path roberta-base   --do_lower_case  --per_gpu_eval_batch_size 1024   --task_name yelp-5   --data_dir  ../dataset/yelp5/     --num_train_epochs 10.0   --logging_steps  300  --dropout_prob 0.1   --output_feature 0   --num_hubo_layers  3  --method  null  --seq_select_prob 0.0  --per_gpu_train_batch_size 64   --max_seq_length 256   --learning_rate 2e-5    --seed  42 `

> roberta-laptop: `python run_sentiment_classifier.py   --do_train   --do_eval   --output_dir  laptop_roberta/    --model_type roberta    --model_name_or_path roberta-base   --do_lower_case  --per_gpu_eval_batch_size 512   --task_name laptop   --data_dir  ../dataset/laptop/     --num_train_epochs 2.0   --logging_steps  50  --dropout_prob 0.0   --output_feature 0   --num_hubo_layers  3  --method  null  --seq_select_prob 0.0  --per_gpu_train_batch_size 16   --max_seq_length 128   --learning_rate 2e-5    --seed  42 `

> roberta-restaurants: `python run_sentiment_classifier.py   --do_train   --do_eval   --output_dir  restaurants_roberta/    --model_type roberta    --model_name_or_path roberta-base   --do_lower_case  --per_gpu_eval_batch_size 512   --task_name restaurants   --data_dir  ../dataset/restaurants/     --num_train_epochs 2.0   --logging_steps  50  --dropout_prob 0.0   --output_feature 0   --num_hubo_layers  3  --method  null  --seq_select_prob 0.0  --per_gpu_train_batch_size 16   --max_seq_length 128   --learning_rate 2e-5    --seed  42 `
--------
>  run TraceNet+XLNet/roberta/glove
> TraceNet+X:  `python run_sentiment_classifier.py   --model_type  xlnet_tracenet   --model_name_or_path ../xlnet_base_cased   --do_train   --do_eval    --output_hidden_states   --output_item_weights   --per_gpu_eval_batch_size 500   --overwrite_output_dir   --output_dir   sst_xlnetTraceNet_5642/    --task_name sst-5    --data_dir ../dataset/SST_5/   --num_hubo_layers  3  --method 'mean'  --proactive_masking    --seq_select_prob  0.2  --dropout_prob 0.3  --output_feature 128  --per_gpu_train_batch_size 16     --max_seq_length 64   --learning_rate 2e-5   --num_train_epochs 10.0   --seed 1 `

> TraceNet+R:  `python run_sentiment_classifier.py   --model_type  roberta_tracenet   --model_name_or_path roberta-base   --do_lower_case  --do_train   --do_eval    --output_hidden_states   --output_item_weights   --per_gpu_eval_batch_size 512    --output_dir   sst_robertaTraceNet/    --task_name sst-5    --data_dir ../dataset/sst5/   --num_hubo_layers  3  --method mean  --proactive_masking    --seq_select_prob  0.3  --dropout_prob 0.1  --output_feature 512  --per_gpu_train_batch_size 16   --max_seq_length 128   --weight_decay 0.0   --adam_epsilon  1e-6   --learning_rate 2e-5   --num_train_epochs 10.0  --logging_steps 300  --seed 42 `

> TraceNet+R+yelp:  `python run_sentiment_classifier.py   --model_type  roberta_tracenet   --model_name_or_path roberta-base  --do_lower_case  --do_train   --do_eval    --output_hidden_states   --output_item_weights   --per_gpu_eval_batch_size 512    --output_dir   yelp_robertaTraceNet/    --task_name yelp-5    --data_dir ../dataset/yelp5/   --num_hubo_layers  3  --method mean  --proactive_masking    --seq_select_prob  0.3  --dropout_prob 0.1  --output_feature 512  --per_gpu_train_batch_size 64   --max_seq_length 256   --weight_decay 0.0   --adam_epsilon  1e-6   --learning_rate 2e-5   --num_train_epochs 10.0  --logging_steps 300  --seed 42 `

> TraceNet+R+laptop:  `python run_sentiment_classifier.py   --model_type  roberta_tracenet   --model_name_or_path roberta-base  --do_lower_case  --do_train   --do_eval    --output_hidden_states   --output_item_weights   --per_gpu_eval_batch_size 512    --output_dir   laptop_robertaTraceNet/    --task_name laptop    --data_dir ../dataset/laptop/   --num_hubo_layers  3  --method mean  --proactive_masking    --seq_select_prob  0.3  --dropout_prob 0.1  --output_feature 512  --per_gpu_train_batch_size 16   --max_seq_length 128   --weight_decay 0.0   --adam_epsilon  1e-6   --learning_rate 2e-5   --num_train_epochs 2.0  --logging_steps 50  --seed 42 `

> TraceNet+R+restaurants:  `python run_sentiment_classifier.py   --model_type  roberta_tracenet   --model_name_or_path roberta-base  --do_lower_case  --do_train   --do_eval    --output_hidden_states   --output_item_weights   --per_gpu_eval_batch_size 512    --output_dir   restaurants_robertaTraceNet/    --task_name restaurants    --data_dir ../dataset/restaurants/   --num_hubo_layers  3  --method mean  --proactive_masking    --seq_select_prob  0.3  --dropout_prob 0.1  --output_feature 512  --per_gpu_train_batch_size 16   --max_seq_length 128   --weight_decay 0.0   --adam_epsilon  1e-6   --learning_rate 2e-5   --num_train_epochs 2.0  --logging_steps 50  --seed 42 `

> TraceNet+G: `python run_glove_input.py  --output_hidden_states   --output_item_weights  --per_gpu_eval_batch_size 64   --per_gpu_train_batch_size 64     --max_seq_length 49  --learning_rate 0.001   --output_feature 50   --dropout_prob 0.2  --num_train_epochs  10  --task sst-5  --weight_decay   0.2   --proactive_masking  --seq_select_prob 0.05  --seed 1 `

#### Evaluation
To evaluate TraceNet-X (attacks) on SST-5, run:
> generate attacks data from sst test set: `python create_GA_test_sst.py`
> and then `python inference.py  --output_dir  sst_xlnet_5533/   --task_name sst-5  --data_dir ../dataset/SST_5/against/  --do_eval    --per_gpu_eval_batch_size 1000  --model_type xlnet   --model_name_or_path ../xlnet_base_cased --max_seq_length 64 --dropout_prob 0.0 --output_feature 0  --against --overwrite_cache`
> or  `python inference.py  --output_dir  sst_xlnetTraceNet_5642/   --task_name sst-5  --data_dir ../dataset/SST_5/against/  --do_eval    --per_gpu_eval_batch_size 1000  --model_type xlnet_tracenet   --model_name_or_path ../xlnet_base_cased --max_seq_length 64  --dropout_prob 0.0   --output_feature 128  --against --overwrite_cache`
> or  `python inference.py  --output_dir  sst_roberta_5669/   --task_name sst-5  --data_dir ../dataset/SST_5/against/  --do_eval    --per_gpu_eval_batch_size 1000  --model_type roberta   --model_name_or_path ../roberta_base_en --max_seq_length 128  --dropout_prob 0.0   --output_feature 0  --against --overwrite_cache`
> or `python inference.py  --output_dir  sst_robertaTraceNet_5787/   --task_name sst-5  --data_dir ../dataset/SST_5/against/  --do_eval    --per_gpu_eval_batch_size 1000  --model_type roberta_tracenet   --model_name_or_path ../roberta_base_en --max_seq_length 128  --dropout_prob 0.0   --output_feature 512  --against --overwrite_cache`

#### Pre-trained Models
* The transformer based pretrained language models we use are Bert (base, uncased), XLNet (base, cased) and RoBerta (base). With [Transformers](https://huggingface.co/models), you can download it by set `model.from_pretrained('bert-base-uncased')`, `model.from_pretrained('xlnet-base-cased')` and `model.from_pretrained('roberta-base')`, respectively. 
* For models trained by our methods and finetuned on SST-5 dataset, you can download from here to inferece.


#### Results
The following is the performance of baselines and TraceNet on SST-5 test set, for detailed parameters and results output, please refer to log.sst5 file:

| model     | KIMCNN rand      |     KIMCNN static |   KIMCNN nonstatic   |  KIMCNN mulch   |  LSTM   |  BiLSTM   |  GT-LSTM(rand)   |GT-LSTM(glove)   |
| :-------- | :-------- | --------:| :------: | :------: | :------: | :------: |:------: |:------: |
| reimplement results    | 40.1    |   44.6 |  44.5  |  43.1  |  44.7  |  43.5  |  39.38  |  47.6  |
| reported results    | 45    |   45.5 |  48  |  47.4  |  46.4(1.1)  |  49.1(1.0)  | None  | None  |

| model      |     Bert |   XLNet   |Roberta      |     TraceNet-X |   TraceNet-R   |  TraceNet-G   |
| :-------- | --------:| :------: |:-------- | --------:| :------: |:------: |
| reimplement results    | 53.07    |   55.33|  56.69  |56.42    |   57.87 |  45.90  |
The results leaderboards of SST-5 dataset on [Papers with Code leaderboards](https://paperswithcode.com/sota) is [here](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained). 


