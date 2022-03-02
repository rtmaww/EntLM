import torch
from transformers import BertForMaskedLM, BertTokenizer, AutoConfig, AutoTokenizer, XLNetLMHeadModel, RobertaForMaskedLM
import torch.nn.functional as F
import json
import os
import argparse

def mask_entity_batch(batch, tokenizer, mask_prob):
    # 只针对当前batch的entity按概率进行mask
    ori_input_ids, input_mask, segment_ids, label_ids, subword_mask, ori_label_ids = batch
    input_ids = ori_input_ids.clone().detach()
    entity_index = ori_label_ids != 0
    entity_mask = torch.full(entity_index.shape, 0.0)
    entity_mask[entity_index] = mask_prob
    entity_prob_mat = torch.bernoulli(entity_mask).bool()
    input_ids[entity_prob_mat] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return [(input_ids, input_mask, segment_ids, label_ids, subword_mask, ori_label_ids), batch]

def collect_label_token_from_LM(model, label_list, support_data_loader, tokenizer, device, filter_list=None, k=3):
    """
    label_list: ['I-PER', 'I-LOC']
    """
    # 不区分BIO标签，实体词的每个subword都统计
    if filter_list==None:
        filter_list = [",", ":", ";", "'", '"', "/", ".", "(", ")", "?", "-", "and", "or", "the", "a","of"]
    label_map = {label[2:]:{} for label in label_list}
    label_map_include_gold = {label[2:]: {} for label in label_list}
    ori_label_map = {index:label for index, label in enumerate(label_list)}
    ori_label_map[-100] = 'O'
    import tqdm
    model.eval()
    with tqdm.tqdm(total=len(support_data_loader)) as pbar:
        for step, batch in enumerate(support_data_loader):

            # batch, ori_batch = mask_entity_batch(batch, tokenizer, 1.0)

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, subword, ori_label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
            logits = torch.topk(F.log_softmax(logits,dim=2), k=k, dim=2).indices
            pbar.update()
            # 根据ori_label_ids中非O的位置，去寻找对应位置的token
            for i in range(label_ids.shape[0]):
                ori_label_batch = label_ids[i]
                pred_top_k = logits[i]
                gold_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                for j in range(len(ori_label_batch)):
                    ori_label = ori_label_map[ori_label_batch[j].item()]
                    gold_token = gold_tokens[j]
                    # gold_token = []
                    if gold_tokens[j] in ['[CLS]', '[SEP]', '[PAD]'] or ori_label == 'O':
                        continue
                    ori_label = ori_label[2:]
                    top_k_token = tokenizer.convert_ids_to_tokens(pred_top_k[j,:])
                    for token in top_k_token:
                        if token not in filter_list:
                            if token not in gold_token:
                                if token not in label_map[ori_label]:
                                    label_map[ori_label][token] = 0
                                label_map[ori_label][token] += 1
                            if token not in label_map_include_gold[ori_label]:
                                label_map_include_gold[ori_label][token] = 0
                            label_map_include_gold[ori_label][token] += 1

    return label_map, label_map_include_gold

def collect_label_token_from_LM_prob(model, label_list, support_data_loader, tokenizer, device, filter_list=None, k=3):
    """
    label_list: ['I-PER', 'I-LOC']
    """
    # 不区分BIO标签，实体词的每个subword都统计
    if filter_list==None:
        filter_list = [",", ":", ";", "'", '"', "/", ".", "(", ")", "?", "-", "and", "or", "the", "a"]
    label_map = {label[2:]:{} for label in label_list}
    ori_label_map = {index:label for index, label in enumerate(label_list)}
    ori_label_map[-100] = 'O'
    import tqdm
    model.eval()
    with tqdm.tqdm(total=len(support_data_loader)) as pbar:
        for step, batch in enumerate(support_data_loader):

            # batch, ori_batch = mask_entity_batch(batch, tokenizer, 1.0)

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, subword, ori_label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
            logits = torch.topk(F.log_softmax(logits,dim=2), k=k, dim=2).indices
            pbar.update()
            # 根据ori_label_ids中非O的位置，去寻找对应位置的token
            for i in range(label_ids.shape[0]):
                ori_label_batch = label_ids[i]
                pred_top_k = logits[i]
                gold_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                for j in range(len(ori_label_batch)):
                    ori_label = ori_label_map[ori_label_batch[j].item()]
                    gold_token = gold_tokens[j]
                    if gold_tokens[j] in ['[CLS]', '[SEP]', '[PAD]'] or ori_label == 'O':
                        continue
                    ori_label = ori_label[2:]
                    top_k_token = tokenizer.convert_ids_to_tokens(pred_top_k[j,:])
                    for token in top_k_token:
                        if token not in filter_list and token not in gold_token:
                            if token not in label_map[ori_label]:
                                label_map[ori_label][token] = 0
                            label_map[ori_label][token] += 1
    return label_map


def collect_label_token_from_data(data_path, label_list, args, dataset='conll', max_seq_length=128, batch_size=64, device='cuda', model_path='/home/jotion/NER/few-shot/bert/models', save_path=None, tokenizer=None, model=None):
    from data_process import NerGeneralProcessor, convert_examples_to_features
    from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
    if model == None:
        if "xlnet" in model_path:
            model = XLNetLMHeadModel.from_pretrained(model_path)
        elif "roberta" in model_path:
            model = RobertaForMaskedLM.from_pretrained(model_path)
        else:
            model = BertForMaskedLM.from_pretrained(model_path)
        model.to(device)
    processor = NerGeneralProcessor()
    print('### loading data ###')
     # 统计高频词时不区分BI 
    train_examples = processor.get_examples(data_path, schema='IO', label_type=args.label_type)
    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    print('### collecting label token ###')
    label_token_map, label_token_map_include_gold = collect_label_token_from_LM(model,
                                                                                label_list, train_dataloader, tokenizer, device)
    if save_path:
        json.dump(label_token_map, open(save_path, 'w'))
        json.dump(label_token_map_include_gold, open(save_path[:-5]+"_include_gold.json", 'w'))
    return label_token_map

def build_label_vocabulary(label_token_map):
    label_map = {}
    label_filter= ['with', 'this', 'that', 'their', 'from']
     # 排序后取每个类别符合过滤标准的最高频词
    for label_name, token_frac_dict in label_token_map.items():
        for token, frac in sorted(token_frac_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
            # 过滤准则，暂时只按照长度过滤, 或许可以换成内部聚类、或者直接top10的embedding相加
            if len(token)>3 and token not in label_filter and "##" not in token:
                label_map[label_name] = token
                label_filter.append(token)
                break
        if label_name and label_name not in label_map:
            label_map[label_name] = label_name
    return label_map

def show_topk_frac(label_token_map, k=10):
    label_map = {}
     # 排序后取每个类别符合过滤标准的最高频词
    for label_name, token_frac_dict in label_token_map.items():
        cnt = 0
        for token, frac in sorted(token_frac_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
            # 过滤准则，暂时只按照长度过滤
            if label_name not in label_map:
                label_map[label_name] = {}
            if len(token)>3 and "##" not in token:
                label_map[label_name][token] = frac
                cnt+=1
            if cnt>k:
                break
    return label_map





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default='',
                        type=str)
    parser.add_argument('--label_list_path',
                        default='',
                        type=str)
    parser.add_argument('--save_dir',
                        default='',
                        type=str)
    parser.add_argument('--model_path',
                        default='bert-base-cased',
                        type=str)
    parser.add_argument('--dataset',
                        default='conll',
                        type=str)
    parser.add_argument('--label_type',
                        default='conll',
                        type=str)
    args = parser.parse_args()
    
    data_path = args.data_path

    label_list = json.load(open(args.label_list_path, 'r'))
    if 'O' not in label_list:
        label_list += ['O']
    # save_path = args.save_path
    print(label_list)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_path = os.path.join(args.save_dir, 'label_frac.json')
    collect_label_token_from_data(data_path, label_list, args, dataset=args.dataset,max_seq_length=128, batch_size=64, device='cuda', model_path=args.model_path, save_path=save_path)
    
    label_frac = json.load(open(save_path, 'r'))
    label_map = build_label_vocabulary(label_frac)
    label_map = {'I-{}'.format(key):value for key,value in label_map.items()}
    print(label_map)
    save_path = os.path.join(args.save_dir, 'label_map.json')
    json.dump(label_map, open(save_path, 'w'))