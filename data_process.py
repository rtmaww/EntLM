import torch
from torch.utils.data import TensorDataset, DataLoader
import random 
from tqdm import tqdm

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, ori_label=None, subword=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.ori_label = ori_label
        self.subword = subword

def readfile(filename, schema='BIO', sep=' '):
    '''
    数据在txt中格式应该为 
    John B-PER
    Wick I-PER
    say O
    若schema为IO, 则会强制将B改为I，若为其他，则正常读取
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                # yield (sentence,label)
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.strip().split()
        sentence.append(splits[0])
        # label.append("O")   ### 为了further pretrain改的
        if schema=='IO' and splits[-1].startswith('B-'):
            label.append('I-{}'.format(splits[-1][2:]))
        else:
            label.append(splits[-1])

    if len(sentence) >0:
        # yield (sentence,label)
        data.append((sentence,label))
        sentence = []
        label = []
    return data

def collect_label_list(data_path, label_type='fine', sep='\t'):
    f = open(data_path)
    label_list = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            continue
        splits = line.strip().split(sep)
        if label_type == 'fine':
            label = splits[-1].split('-')[-1]
        else:
            label = splits[-1].split('-')[0]
        if label not in label_list:
            label_list.append(label)
    return label_list

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file, schema='BIO', sep= ' ', quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file, schema, sep)

    def _create_examples(self, lines, set_type):
        examples = []
        for i,(sentence, label) in tqdm(enumerate(lines), desc='Create {} examples'.format(set_type)):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

class NerGeneralProcessor(DataProcessor):
    """Processor for the general ner data set."""
    def get_examples(self, data_path, schema='IO', sep=' ', data_type='train', label_type='fine'):
        # label_type 为冗余参数
        return self._create_examples(
            self._read_file(data_path, schema=schema, sep=sep), data_type)

    def get_label_map(self, dataset=''):
        """
            Returns a mapping dict from label to label token
        """
        labels_map = {
                    'ontonotes':{"I-CARDINAL": "three", "I-DATE": "years", "I-EVENT": "Christmas", 
                                "I-FAC": "their", "I-GPE": "China", "I-LANGUAGE": "language", "I-LAW": "this", 
                                "I-LOC": "South", "I-MONEY": "millions", "I-NORP": "Arab", "I-ORDINAL": "second", 
                                "I-ORG": "Corporation", "I-PERCENT": "percent", "I-PERSON": "John", "I-PRODUCT": "ship", 
                                "I-QUANTITY": "feet", "I-TIME": "evening", "I-WORK_OF_ART": "with"},
                    'conll': {  'I-LOC':'Australia',
                               'I-PER':'John',
                                'I-ORG':'Company',
                                 'I-MISC':'German'},
                    # 'conll': {'I-LOC': ['Australia', 'Germany', 'England', 'Canada', 'France', 'Italy', 'Belgium', 'United', 'Spain', 'Ireland', 'Argentina'],
                    #           'I-PER': ['John', 'Michael', 'David', 'Smith', 'Paul', 'Thomas', 'Mark', 'Robert', 'Mike', 'Peter', 'Tony'],
                    #           'I-ORG': ['United', 'Corporation', 'National', 'Company', 'Central', 'London', 'Union', 'City', 'Council', 'Boston', 'Corp'],
                    #           'I-MISC': ['German', 'American', 'British', 'Australian', 'former', 'Canadian', 'Championship', 'French', 'European', 'local', 'Turkish']}
                    # 'conll': {}  ## further pretrain
        }
         
        return labels_map[dataset]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list,0)}
    features = []
    for (ex_index,example) in tqdm(enumerate(examples), desc='Examples2Features'):
        # 此处textlist 和 labellist 应当等长
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            # 获得bpe
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]

            # 只保留第一个bpe的label, 但是要记录第一个bpe的位置 0代表要mask, 1代表有用
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    labels.append('O') # TODO 
                    valid.append(0)
                    label_mask.append(0)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,0)
        label_mask.insert(0,0)
        # label_ids.append(label_map["[CLS]"])
        label_ids.append(-100)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
            else:
                print(labels[i])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(0)
        label_mask.append(0)
        label_ids.append(-100)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        # with open('tmp1.txt','a') as f1:
        #     f1.write(str(input_ids))
        input_mask = [1] * len(input_ids)
        
        # label_mask += [1] * len(label_ids)
        assert len(label_mask) == len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-100)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(-100)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features

def convert_examples_to_features_lm(examples, label_map, max_seq_length, tokenizer, subword_map):
    """
    label_map = {'I-PER':'person' ......}
    
    """
    ori_label_map = {key:idx+1 for idx, key in enumerate(label_map.keys())}
    ori_label_map['O'] = 0
    features = []
    for ex_index, example in tqdm(enumerate(examples), desc='Examples2Features'):
        # 一个句子
        text_list = example.text_a.split(' ')
        label_list = example.label

        subword_list = []
        label_token_list = []

        subword_mask = []
        ori_label_list = []

        for label, token in zip(label_list, text_list):
            subwords = tokenizer.tokenize(token)
            for i in range(len(subwords)):
                subword_list.append(subwords[i])
                if i == 0:
                    # 首个subword
                    subword_mask.append(1)
                    ori_label_list.append(label)
                    if label == 'O':
                        label_token_list.append(tokenizer.convert_tokens_to_ids(subwords[i]))     ### 预测自己
                    else:
                        label_token_list.append(tokenizer.convert_tokens_to_ids(label_map[label]))  ### 预测标签
                        # label_token_list.append(tokenizer.convert_tokens_to_ids(subwords[i]))   ### 预测自己
                else:
                    # 其余subword
                    subword_mask.append(0)
                    # ori_label_list.append('O')
                    ori_label_list.append(label)
                    # subword 三个方案：目标是自己，目标是unused，目标是label token
                    
                    # label_token_list.append(subwords[i])  # 目标自己
                    if label == 'O':
                        label_token_list.append(tokenizer.convert_tokens_to_ids(subwords[i]))   ### 预测自己
                    else:
                        # label_token_list.append(subword_map[label])  # 目标unused 预测subword标签
                        label_token_list.append(tokenizer.convert_tokens_to_ids(label_map[label]))  # 目标label token 预测标签
                        # label_token_list.append(tokenizer.convert_tokens_to_ids(subwords[i]))   ### 预测自己
                        # label_token_list.append(-100)   ### 不算loss

    

        assert len(subword_list) == len(label_token_list)
        assert len(subword_list) == len(subword_mask)
        assert len(subword_list) == len(ori_label_list)
        
        # 最大长度截断
        if len(label_token_list) >= max_seq_length-1:
            subword_list = subword_list[:max_seq_length-2]
            label_token_list = label_token_list[:max_seq_length-2]
            subword_mask = subword_mask[:max_seq_length-2]
            ori_label_list = ori_label_list[:max_seq_length-2]
        
        # 添加[CLS]和[SEP]标识
        subword_list.insert(0,'[CLS]')
        subword_list.append('[SEP]')

        label_token_list.insert(0, tokenizer.convert_tokens_to_ids('[CLS]'))
        label_token_list.append(tokenizer.convert_tokens_to_ids('[SEP]'))

        # [CLS] 和 [SEP]subword设置为0，目的是在evaluate的时候不进行计算
        subword_mask.insert(0, 0)
        subword_mask.append(0)

        ori_label_list.insert(0, 'O')
        ori_label_list.append('O')
        
        input_ids = tokenizer.convert_tokens_to_ids(subword_list)
        # label_ids = tokenizer.convert_tokens_to_ids(label_token_list)
        label_ids = label_token_list
        # 不计算[CLS]和[SEP]
        # label_ids[0] = -100
        # label_ids[-1] = -100

        ori_label_ids = [ori_label_map[label] for label in ori_label_list]

        segment_ids = [0]*len(subword_list)
        input_mask = [1]*len(subword_list)
        
        # padding到最大长度
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            subword_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-100)
            ori_label_ids.append(0)

        assert len(input_ids) == len(label_ids)
        assert len(input_ids) == len(input_mask)
        assert len(input_ids) == len(segment_ids)
        assert len(input_ids) == len(subword_mask)
        assert len(input_ids) == len(ori_label_ids)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              subword=subword_mask,
                              ori_label=ori_label_ids))
    return features

def get_data_loader(train_examples, label_list, max_seq_length, tokenizer, batch_size, sampler):
    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
    train_sampler = sampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return dataloader

def get_data_loader_lm(train_examples, label_map, max_seq_length, tokenizer, batch_size, sampler, subword_map):
    train_features = convert_examples_to_features_lm(
        train_examples, label_map, max_seq_length, tokenizer,subword_map)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_ori_label_ids = torch.tensor([f.ori_label for f in train_features], dtype=torch.long)
    subword = torch.tensor([f.subword for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, subword, all_ori_label_ids)
    train_sampler = sampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return dataloader
