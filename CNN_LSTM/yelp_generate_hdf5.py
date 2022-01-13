import numpy as np
import h5py


def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(' ')
            word = tmp[0]
            if word in vocab:
                vec = list(map(float, tmp[1:]))
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
    max_len = 0
    with open(file, encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            words = text.split()
            if len(words) > max_len:
                max_len = len(words)
            text_list.append(words)
            label_list.append(label)
    print("max sequence lengths: ", max_len)
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
        ids = [word_to_idx[w] for w in words][:sequence_length]
        ids = ids + [0] * (sequence_length - len(ids))
        text_id_list.append(ids)
        label_id_list.append(label_idx[label])
    return np.array(text_id_list), np.array(label_id_list)


def write_yelp_glove(target_file, vocab):
    source_file = '../dataset/glove.840B.300d.txt'
    fw = open(target_file, 'w', encoding='utf-8')
    with open(source_file, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(' ')
            word = tmp[0]
            if word in vocab:
                fw.write(line)


if __name__ == "__main__":
    dataset = 'yelp-5'
    sequence_length = 512
    train_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.train"
    dev_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.dev"
    test_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.test"
    w2v_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.glove.to.tranceNet"

    h5_file = "../dataset/Yelp_5/yelp_review_full_csv/yelp5.hdf5"
    label_idx = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}

    train, train_label = load_data(train_file)
    dev, dev_label = load_data(dev_file)
    test, test_label = load_data(test_file)
    word_to_idx = get_vocab(train, dev, test)

    write_yelp_glove(w2v_file, word_to_idx)

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