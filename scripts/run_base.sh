# Run the Bert-tagger baseline

for j in {1,2,3,4}
#j=2
do
FILE_PATH=dataset/conll/5shot/
FILE_NAME=$FILE_PATH${j}.json

for i in {1..4}
do

echo "---------------------------------------Training with file ${FILE_NAME}, round $i----------------------------------------"
echo ""


CUDA_VISIBLE_DEVICES=4 python3 run_ner_no_trainer.py \
  --model_name_or_path pretrained/bert-base-cased \
  --train_file $FILE_NAME \
  --validation_file dataset/conll/test.json \
  --output_dir models/test-ner \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 4 \
  --label_list dataset/conll/labels.txt \
  --return_entity_level_metrics \
  --label_schema IO \


done
done
