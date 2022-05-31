import random
import json

# random.seed(1348)
def sample_data(data_path, output_path, k=10):
    with open(data_path, 'r') as f:
        few_shot_data = []
        label_cnt_dict = {}
        data = f.readlines()
        random.shuffle(data)
        for row in data:
            item = eval(row)
            label = item['label']
            if len(label) <= 10:
                continue
            is_add = True
            temp_cnt_dict = {}
            for l in label:
                if l.startswith('B-'):
                    if l not in temp_cnt_dict:
                        temp_cnt_dict[l] = 0
                    temp_cnt_dict[l] += 1
                    if l not in label_cnt_dict:
                        label_cnt_dict[l] = 0
                    if label_cnt_dict[l] + temp_cnt_dict[l] > k:
                        is_add = False
            if len(temp_cnt_dict)==0:
                is_add=False
            if is_add:
                few_shot_data.append(item)
                for key in temp_cnt_dict.keys():
                    label_cnt_dict[key] += temp_cnt_dict[key]
    with open(output_path, 'w') as wf:
        for row in few_shot_data:
            json.dump(row, wf)
            wf.write('\n')
    return label_cnt_dict



if __name__ == '__main__':
    import os
    
    for k in [5]:
        for i in range(1,4):
            path = f"dataset/conll/{k}shot"
            if not os.path.exists(path):
                os.mkdir(path)

            sample_data(f"dataset/conll/train.json", f"{path}/{i}.json", k=k)