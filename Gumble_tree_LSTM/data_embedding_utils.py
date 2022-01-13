import numpy as np
import torch, random


def get_vocab(path):
    word_to_idx = {'UNK': 0, 'PAD': 1}
    idx = 2
    with open(path, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split('\t')[-1]
            words = words.split()
            for w in words:
                if w not in word_to_idx:
                    word_to_idx[w] = idx
                    idx += 1
    return word_to_idx


def load_vectors(file):
    with open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        data = {}
        for line in f:
            tokens = line.strip().split(' ')
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            data[word] = vec
    return data


def compute_embed(word_to_idx, w2v_file):
    w2v = load_vectors(file=w2v_file)
    V = len(word_to_idx)
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, 300))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec # padding word is positioned at index 0
    return embed


def get_batch_data(args, data_file, vocab_dic, batch_size):
    all_data = []
    label_dic_sst = {'__label__1': 0, '__label__2': 1, '__label__3': 2, '__label__4':3, '__label__5': 4}
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            label, sentence = line.strip().split('\t')
            words = [vocab_dic.get(w, 0) for w in sentence.split()]
            line_dic = {}
            line_dic['text'] = words[:args.max_seq_length]
            if args.task == 'yelp-5':
                line_dic['label'] = int(label) -1
            elif args.task == 'sst-5':
                line_dic['label'] = label_dic_sst[label]
            line_dic['length'] = len(words[:args.max_seq_length])
            all_data.append(line_dic)
    random.shuffle(all_data)
    dataBuckt = []
    for start, end in zip(range(0, len(all_data), batch_size), range(batch_size, len(all_data), batch_size)):
        batchData = all_data[start: end]
        dataBuckt.append(batchData)
    newDataBuckt = []
    for idx, batch in enumerate(dataBuckt):
        batch_tmp = {"length": [], "text": [], "label": [], "iterations": idx+1}
        for data in batch:
            batch_tmp["length"].append(data['length'])
            batch_tmp["text"].append(data['text'])
            batch_tmp["label"].append(data['label'])
        max_len = args.max_seq_length
        batch_tmp["text"] = torch.LongTensor([x + [1] * (max_len-len(x)) for x in batch_tmp["text"]]) # pad
        batch_tmp["length"] = torch.LongTensor(batch_tmp["length"])
        batch_tmp["label"] = torch.LongTensor(batch_tmp["label"])
        newDataBuckt.append(batch_tmp)
    return newDataBuckt
