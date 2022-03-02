
from numpy import delete
import torch
import torch.nn.functional as F

def get_label_from_label_token(token_list, label_map, mode='IO'):
    """
        label_map = {'person':'PER',
                     'location': 'LOC'
                    ...
                    ...
                    }
    """
    label_list = []
    past_label = ''
    for i in range(len(token_list)):
        token = token_list[i]
        if token in label_map.keys():
            current_label = label_map[token]
            if mode == 'BIO' and current_label != past_label:
                label_list.append('B-{}'.format(current_label))
            else:
                label_list.append('I-{}'.format(current_label))
        else:
            current_label = 'O'
            label_list.append(current_label)
        past_label = current_label
    return label_list

def filter_item(item_list, subword_mask, input_mask):
    # 只保留每个词第一个subword和非[PAD][CLS][SEP]部分
    clean_item_list = []
    for item, not_subword, not_mask in zip(item_list, subword_mask, input_mask):
        if not_subword==0:
            continue
        if not_mask==0:
            break
        clean_item_list.append(item)
    return clean_item_list

def get_label_token_from_topk(pred_ids_topk, tokenizer, label_map, seq_len=None): ### top k预测，只要label token在top k内就认为预测了该label，选排在最前的
    if seq_len == None:
        seq_len = pred_ids_topk.shape[0]
    label_list = label_map.values()
    pred_token = []
    for i in range(seq_len):
        # 当前位置topk的词
        top_k_token = tokenizer.convert_ids_to_tokens(pred_ids_topk[i][:])
        for token in top_k_token:
            if token in label_list:
                pred_token.append(token)
                break
        # topk中无label token，则取概率最高的词
        if len(pred_token) == i:
            pred_token.append(top_k_token[0])
    assert len(pred_token) == seq_len
    return pred_token

def get_label_from_ids(ori_label_ids, subword, input_mask, label_map):
    ori_label_map = {key:idx+1 for idx, key in enumerate(label_map.keys())}
    ids_label_map = {value:key for key,value in ori_label_map.items()}
    ids_label_map[0] = 'O'
    batch_size = ori_label_ids.shape[0]
    label_list = []
    for i in range(batch_size):
        batch_label = []
        # delete_idx = []
        for j in range(len(ori_label_ids[i])):
            # 去除CLS和SEP
            if subword[i][j] == 0:
                continue
            if input_mask[i][j] == 0:
                break
            batch_label.append(ids_label_map[ori_label_ids[i][j].item()])
        label_list.append(batch_label)
    return label_list

def get_label_from_logits(logits, label_ids, input_ids, subword, input_mask, tokenizer, label_map, k=1, mode='IO', print_topk=0):

    pred_ids_topk = torch.topk(logits, k=k, dim=2).indices

    if print_topk > 0:
        pred_value_top5, pred_ids_top5 = torch.topk(logits, k=print_topk, dim=2)


    pred_labels = []
    pred_tokens = []
    gold_tokens = []
    pred_tokens_top5 = []
    batch_size = label_ids.shape[0]
    for i in range(batch_size):
        gold_token = tokenizer.convert_ids_to_tokens(input_ids[i])
        pred_token = get_label_token_from_topk(pred_ids_topk[i], tokenizer, label_map)
        
        gold_token = filter_item(gold_token, subword_mask=subword[i], input_mask=input_mask[i])
        pred_token = filter_item(pred_token, subword_mask=subword[i], input_mask=input_mask[i])

        if print_topk > 0:
            pred_tokens_top5_ = [((tokenizer.convert_ids_to_tokens(word_ids)), values) for word_ids, values in zip(pred_ids_top5[i], pred_value_top5[i])]
            pred_tokens_top5.append(filter_item(pred_tokens_top5_,
                                            subword_mask=subword[i], input_mask=input_mask[i]))
        # pred_token_label = tokenizer.convert_ids_to_tokens(pred_ids[i])
        
        # 相当于没有I标签，只有B标签
        reverse_label_map = {value:key[2:] for key,value in label_map.items()}
        pred_label = get_label_from_label_token(pred_token, reverse_label_map, mode)

        assert len(gold_token) == len(pred_token)
        assert len(gold_token) == len(pred_label)

        gold_tokens.append(gold_token)
        pred_tokens.append(pred_token)
        pred_labels.append(pred_label)

    if print_topk > 0:
        return pred_labels, pred_tokens, gold_tokens, pred_tokens_top5
    else:
        return pred_labels, pred_tokens, gold_tokens




if __name__ == '__main__':
    label_map = {'B-LOC':'location', 'I-LOC':'location',
                'B-PER':'person', 'I-PER':'person',
                'B-ORG':'organization', 'I-ORG':'organization',
                'B-MISC':'[unused90]', 'I-MISC':'[unused90]'}
    # label_map = {'B-LOC':'location', 'I-LOC':'[unused50]',
    #             'B-PER':'person', 'I-PER':'[unused51]',
    #             'B-ORG':'organization', 'I-ORG':'[unused52]',
    #             'B-MISC':'[unused90]', 'I-MISC':'[unused91]'}
    # label_map = {'B-LOC':'location', 'I-LOC':'location',
    #             'B-PER':'person', 'I-PER':'person',
    #             'B-ORG':'organization', 'I-ORG':'organization',
    #             'B-MISC':'[unused70]', 'I-MISC':'[unused70]'}
    # label_map = {'B-LOC':'location', 'I-LOC':'<<LOC_subword>>',
    #             'B-PER':'person', 'I-PER':'<<PER_subword>>',
    #             'B-ORG':'organization', 'I-ORG':'<<ORG_subword>>',
    #             'B-MISC':'[unused70]', 'I-MISC':'<<MISC_subword>>'}
    new_label_map = {value:key for key, value in label_map.items()}

    sentence = ['[CLS]', 'person', '<<PER_subword>>', 'person', '<<PER_subword>>', '[SEP]', '[PAD]', '[PAD]']
    subword = [1,1,0,1,0,1,0,0]
    input_mask = [1]*len(sentence)
    input_mask.extend([0, 0])
    label_map = []
    print(get_label_from_label_token(sentence, label_map=new_label_map, input_mask=input_mask, subword=subword))