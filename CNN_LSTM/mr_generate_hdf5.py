import numpy as np
import h5py


def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split()
            word, vec = tmp[0], list(map(float, tmp[1:]))
            if word in vocab:
                word_vecs[word] = vec
    return word_vecs


def get_vocab(train, dev, test):
    word_to_idx = {'PAD': 0}
    dataset = train + dev + test
    idx = 1
    for words in dataset:
        for w in words:
            if not w in word_to_idx:
                word_to_idx[w] = idx
                idx += 1
    return word_to_idx


def load_data(file):
    text_list, label_list = [], []
    with open(file, encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            text_list.append(text.split())
            label_list.append(label)
    print('max length: ', max([len(x) for x in text_list]))
    return text_list, label_list


def compute_embed(V, w2v):
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, 300))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec # padding word is positioned at index 0
    return embed


def convert_word_to_id(word_to_idx, text_list, label_list, label_idx, sequence_length):
    text_id_list, label_id_list = [], []
    for words, label in zip(text_list, label_list):
        ids = [word_to_idx[w] for w in words]
        ids = ids + [0] * (sequence_length - len(ids))
        text_id_list.append(ids)
        label_id_list.append(label_idx[label])
    return np.array(text_id_list), np.array(label_id_list)


def write_glove(target_file, source_file, vocab):
    fw = open(target_file, 'w', encoding='utf-8')
    with open(source_file, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(' ')
            word = tmp[0]
            if word in vocab:
                fw.write(line)


if __name__ == "__main__":
    dataset = 'sst-5'
    sequence_length = 128
    source_file = '../dataset/glove.840B.300d.txt'

    train_file = "../dataset/MR_SentiLARE/MR-2.train"
    dev_file = "../dataset/MR_SentiLARE/MR-2.dev"
    test_file = "../dataset/MR_SentiLARE/MR-2.test"
    w2v_file = "../dataset/MR_SentiLARE/MR2.glove.to.TraceNet"
    h5_file = "../dataset/MR_SentiLARE/mr2.hdf5"
    label_idx = {'0': 0, '1': 1}

    train, train_label = load_data(train_file)
    dev, dev_label = load_data(dev_file)
    test, test_label = load_data(test_file)
    word_to_idx = get_vocab(train, dev, test)

    write_glove(w2v_file, source_file, word_to_idx)
    train, train_label = convert_word_to_id(word_to_idx, train, train_label, label_idx, sequence_length)
    dev, dev_label = convert_word_to_id(word_to_idx, dev, dev_label, label_idx, sequence_length)
    test, test_label = convert_word_to_id(word_to_idx, test, test_label, label_idx, sequence_length)

    w2v = load_bin_vec(w2v_file, word_to_idx)
    V = len(word_to_idx)
    print('Vocab size:', V)
    embed_w2v = compute_embed(V, w2v)
    print('train size:', train.shape)

    with h5py.File(h5_file, "w") as f:
        f["w2v"] = np.array(embed_w2v)
        f['train'] = train
        f['train_label'] = train_label
        f['dev'] = dev
        f['dev_label'] = dev_label
        f['test'] = test
        f['test_label'] = test_label