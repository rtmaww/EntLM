# Run few-shot experiments of EntLM on conll

for j in {1,2,3,4}
do

FILE_PATH=dataset/conll/5shot/ # 10shot/20shot/50shot
FILE_NAME=$FILE_PATH${j}.json


for i in {1..4}
do

echo ""
echo "---------------------------------------Training with file ${FILE_NAME}, round $i----------------------------------------"
echo ""


CUDA_VISIBLE_DEVICES=1 python train_transformer.py --train_file $FILE_NAME \
                                                   --validation_file dataset/conll/test.json \
                                                   --model_name_or_path pretrained/bert-base-cased \
                                                   --per_device_train_batch_size 4 \
                                                   --learning_rate 1e-4 \
                                                   --return_entity_level_metrics \
                                                   --label_schema IO \
                                                   --eval_label_schema IO \
                                                   --output_dir models/conll-fewshot-test \
                                                   --label_map_path dataset/conll/label_map_timesup_ratio0.6_multitoken_top6.json \
#                                                   --do_crf \
#                                                   --crf_raw_path dataset/conll/distant_data/


done
done