import json
import collections
import argparse
import os


def show_topk_frac(label_token_map, k=2, filter_ratio=0.8, lm_entity_freq=None):


    entity_freq, data_label_token_map = count_entity_freq(args.raw_data_file)

    label_map = {}
    # get the top k word after sorted
    for label_name, token_frac_dict in label_token_map.items():

        cnt = 0

        if args.sort_method == "timesup":
            if not args.ignore_single_appear:
                ## extend the LM dict with data dict, extened token's freq is 1
                for token in data_label_token_map[label_name].keys():
                    if token not in token_frac_dict:
                        token_frac_dict[token] = 1
                sort_key = lambda x: x[1] * entity_freq[x[0]].get(label_name, 1.)
            else:
                sort_key = lambda x: x[1] * entity_freq[x[0]].get(label_name, 0.)
        elif args.sort_method == "data":
            token_frac_dict = data_label_token_map[label_name]
            sort_key = lambda x: x[1]
        elif args.sort_method == "LM":
            sort_key = lambda x: x[1]


        for token, frac in sorted(token_frac_dict.items(), key = sort_key, reverse=True):

            if label_name not in label_map:
                label_map[label_name] = {}
            if len(token)>1 and token in entity_freq and entity_freq[token]: #and "##" not in token
                entity_label_ratio = entity_freq[token].get(label_name, 0) / sum(entity_freq[token].values())
                if entity_label_ratio > filter_ratio:

                    label_map[label_name][token] = (frac, entity_freq[token])
                    cnt+=1
            if cnt>=k:
                break
    return label_map



def filter_is_overlap(token, label_name, label_filter, entity_label_map):
    if len(token)>3 and token not in label_filter and '##' not in token:
        for key, value in entity_label_map.items():
            if key==label_name:
                continue
            if token in value:
                return False
        return True
    else:
        return False

def collect_entity_token(data_path):
    label_map = {}
    with open(data_path, 'r') as f:
        data = f.readlines()
        for row in data:
            item = row.strip()
            if item!='' and item != '-DOCSTART- -X- -X- O':
                splits = item.split()
                token = splits[0]
                label = splits[-1]
                if label != 'O':
                    # 是否区分BI
                    label = label[2:]
                    if label not in label_map:
                        label_map[label] = []
                    if token not in label_map[label]:
                        label_map[label].append(token)
    return label_map


def count_entity_freq(data_path):

    entity_freq = collections.defaultdict(dict)
    label_map = collections.defaultdict(dict)
    with open(data_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line) < 2 or "-DOCSTART-" in line:
            continue
        line = line.strip().split()
        word = line[0]
        label = line[-1]

        if label != "O":
            label = label[2:]
        entity_freq[word][label] = entity_freq[word].get(label, 0) + 1
        label_map[label][word] = label_map[label].get(word, 0) + 1

    return entity_freq, label_map


def count_entity_freq_roberta(data_path):

    entity_freq = collections.defaultdict(dict)
    label_map = collections.defaultdict(dict)
    with open(data_path, 'r') as f:
        lines = f.readlines()

    first = True
    for line in lines:
        if len(line) < 2 or "-DOCSTART-" in line:
            first = True
            continue
        line = line.strip().split()
        word = line[0]
        label = line[-1]

        if not first:
            word = "\u0120"+word

        if label != "O":
            label = label[2:]
        entity_freq[word][label] = entity_freq[word].get(label, 0) + 1
        label_map[label][word] = label_map[label].get(word, 0) + 1

        first = False
    return entity_freq, label_map



def get_lm_entity_freq(label_frac):
    entity_freq = collections.defaultdict(dict)
    for label, token_frac_dict in label_frac.items():
        for token, freq in token_frac_dict.items():
            entity_freq[token][label] = freq

    return entity_freq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_frac_file',
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument('--raw_data_file',
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument('--output_dir',
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument('--filter_ratio',
                        default=0.6,
                        type=float)
    parser.add_argument('--top_k_num',
                        default=6,
                        type=int)
    parser.add_argument("--sort_method",
                        default="timesup",
                        choices=["timesup", "LM", "data"],
                        type=str)
    parser.add_argument("--multitoken",
                        action="store_true")
    parser.add_argument("--ignore_single_appear",
                        action="store_true")
    parser.add_argument("--comment",
                        default="",
                        type=str)

    args = parser.parse_args()

    with open(args.label_frac_file, "r") as f:
        label_frac = json.load(f)

    entity_freq = get_lm_entity_freq(label_frac)
    # entity_freq = None

    print("-----------", args.label_frac_file, "----------")
    label_map = show_topk_frac(label_frac, filter_ratio=args.filter_ratio, k=args.top_k_num, lm_entity_freq=entity_freq)
    for label, tokens in label_map.items():
        print(label, tokens)

    if args.multitoken:
        label_map_output = {"I-"+label:list(tokens.keys()) for label, tokens in label_map.items()}
    else:
        label_map_output = {"I-"+label: list(tokens.keys())[0] if len(tokens.keys()) > 0 else "" for label, tokens in label_map.items()}

    print(label_map_output)

    multitoken_term = "_multitoken" if args.multitoken else ""
    file_name = f"label_map_{args.sort_method}_ratio{args.filter_ratio}{multitoken_term}_top{args.top_k_num}{args.comment}.json"

    output_file = os.path.join(args.output_dir, file_name)
    print("Dumping label_map to ", output_file)
    with open(output_file, "w") as f:
        json.dump(label_map_output, f)

