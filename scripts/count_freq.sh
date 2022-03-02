export OBJ_PATH=dataset/conll/

echo "------- Building label words to $OBJ_PATH ---------"


# 1. Get the label_word_frequency details (label_frac.json) from LM outputs
#CUDA_VISIBLE_DEVICES=0 python language_model.py --data_path ${OBJ_PATH}/distant_data/train.txt \
#                                                --label_list_path ${OBJ_PATH}/label_list.json \
#                                                --save_dir ${OBJ_PATH}/ \
#                                                --model_path pretrained/bert-base-cased/


# 2. Get the label words
python get_topk.py --label_frac_file ${OBJ_PATH}label_frac.json \
                   --raw_data_file ${OBJ_PATH}/distant_data/train.txt \
                   --output_dir ${OBJ_PATH} \
                   --filter_ratio 0.6 \
                   --sort_method timesup \
                   --top_k_num 6 \
                   --multitoken \


