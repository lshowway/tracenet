import random

import pandas as pd

train, dev, test = [], [], []

all_samples = []
# read .csv file
with open('train.csv', encoding='utf-8') as f:
    t = pd.read_csv(f, header=None, names=None)
    all_samples = t.values.tolist()
random.shuffle(all_samples)

train = all_samples[:33000]


with open('Yelp5.sample.train', 'w', encoding='utf-8') as f:
    for line in train:
        f.write(str(line[0]) + '\t' + line[1] + '\n')


with open('test.csv', encoding='utf-8') as f:
    t = pd.read_csv(f, header=None, names=None)
    all_samples = t.values.tolist()
random.shuffle(all_samples)
dev = all_samples[:2500]
test = all_samples[2500: 2500+2500]

with open('Yelp5.sample.dev', 'w', encoding='utf-8') as f:
    for line in dev:
        f.write(str(line[0]) + '\t' + line[1] + '\n')

with open('Yelp5.sample.test', 'w', encoding='utf-8') as f:
    for line in test:
        f.write(str(line[0]) + '\t' + line[1] + '\n')
