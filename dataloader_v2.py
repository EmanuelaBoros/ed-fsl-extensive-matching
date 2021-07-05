from itertools import islice
import collections
import random
import numpy as np
import torch
from tqdm import tqdm
import embeddings as emb
import os
import pickle

def merge(d1, d2):
    for id, sample in d1.items():
        sample.update(d2[id])
    return d1


# def get_labels():
#    return {
#        'O': 0,
#        'Business': 1,
#        'Conflict': 2,
#        'Contact': 3,
#        'Life': 4,
#        'Movement': 5,
#        'Justice': 6,
#        'Transaction': 7,
#        'Personnel': 8
#    }

def get_labels():
    return {
        'O': 0,
        'Execute': 1,
        'Meet': 2,
        'Sentence': 3,
        'Appeal': 4,
        'Transport': 5,
        'End_Org': 6,
        'Acquit': 7,
        'Sue': 8,
        'Attack': 9,
        'Pardon': 10,
        'Demonstrate': 11,
        'Start_Position': 12,
        'Die': 13,
        'Injure': 14,
        'Transfer_Ownership': 15,
        'Marry': 16,
        'Phone_Write': 17,
        'Charge_Indict': 18,
        'Declare_Bankruptcy': 19,
        'Convict': 20,
        'Elect': 21,
        'Nominate': 22,
        'Arrest_Jail': 23,
        'Merge_Org': 24,
        'Be_Born': 25,
        'Transfer_Money': 26,
        'Release_Parole': 27,
        'Fine': 28,
        'Trial_Hearing': 29,
        'Divorce': 30,
        'Extradite': 31,
        'Start_Org': 32,
        'End_Position': 33}


TYPES = {
    "Business": [
        "Start-Org",
        "Merge-Org",
        "Declare-Bankruptcy",
        "End-Org"],
    "Conflict": [
        "Attack",
        "Demonstrate"],
    "Contact": [
        "Meet",
        "Phone-Write"],
    "Life": [
        "Be-Born",
        "Marry",
        "Injure",
        "Divorce",
        "Die"],
    "Movement": ["Transport"],
    "Justice": [
        "Arrest-Jail",
        "Release-Parole",
        "Trial-Hearing",
        "Charge-Indict",
        "Sue",
        "Convict",
        "Sentence",
        "Fine",
        "Execute",
        "Extradite",
        "Acquit",
        "Appeal",
        "Pardon"],
    "Transaction": [
        "Transfer-Ownership",
        "Transfer-Money"],
    "Personnel": [
        "Start-Position",
        "End-Position",
        "Nominate",
        "Elect"]}

DEFAULT_FEATURES = ('indices',
                    'dist',
                    'length',
                    'mask',
                    'anchor_index')
GCN_FEATURES = ('indices',
                'dist',
                'length',
                'mask',
                'anchor_index',
                'adj_arc_in',
                'adj_lab_in',
                # 'adj_arc_out',
                # 'adj_lab_out',
                'adj_mask_in',
                'adj_mask_out',
                'adj_mask_loop',
                'mner'
                )


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert isinstance(item_list, list)
    dico = {}
    for item in item_list:
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1
    return dico


def word_mapping(words):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    dico = create_dico(words)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000

#    dico = {k: v for k, v in dico.items() if v >= 3} #removed thresh
    
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def sliding_window(seq, n=5):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    slices = []
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        slices.append(result)
    for elem in it:
        result = result[1:] + (elem,)
        slices.append(result)

    return slices

def load_embeddings(type_embeddings='glove'):
    if 'glove' in type_embeddings:
        embeddings_model = emb.fetch_GloVe()
    elif 'google' in type_embeddings:
        embeddings_model = emb.fetch_SG_GoogleNews()
    elif 'fast' in type_embeddings:
        embeddings_model = emb.fetch_FastText()
    elif 'number' in type_embeddings:
        embeddings_model = emb.fetch_conceptnet_numberbatch()
    return embeddings_model

import spacy

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
    
nlp = spacy.load("en_core_web_lg")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

import string
punctuation = list(string.punctuation)

def load_ace_dataset(options):
    test_type = options.test_type
    print('test_type', test_type)

    label2index = get_labels()
    
    if os.path.exists('data/word_data_' + str(options.embedding) + '.pkl'):
        with open('data/word_data_' + str(options.embedding) + '.pkl', 'rb') as f:
            word_data = pickle.load(f)
        with open('data/word_embeds_' + str(options.embedding) + '.pkl', 'rb') as f:
            word_embeds = pickle.load(f)
    else:
    
        sentences = []
        for path in [
                ('train', os.path.join(options.data_path, 'train.tsv')),
                ('dev', os.path.join(options.data_path, 'dev.tsv')),
                ('test', os.path.join(options.data_path, 'test.tsv'))]:
            print('Reading', path[0])
            with open(path[1], 'r') as f:
                set_sentences = f.read().split('\n\n')
                print('---', len(set_sentences), 'sentences')
                sentences += set_sentences  # TODO: they say in the paper that they concat all data
    
        word_dim = 300
        # TODO: for now, just random embeddings
            
        def get_event_type(label):  # transform the event subtypes labels into event types
            if len(label) == 1:
                return label
            for key in TYPES.keys():
                subtypes = TYPES[key]
                for subtype in subtypes:
                    if label[2:] in subtype.replace('-', '_'):
                        return subtype
            return label
    
        all_words = []
        word_data = []
        unknown_words = 0
        count = 0
        MAX_DIST = 0
        MIN_DIST = 10000
        for sentence in tqdm(sentences, total=len(sentences)):
            if 'DOCSTART' in sentence:
                continue
            words = [x.split('\t')[0].strip()
                     for x in sentence.split('\n') if len(x.split('\t')) > 1]
            labels = [get_event_type(x.split('\t')[1].strip())
                      for x in sentence.split('\n') if len(x.split('\t')) > 1]
            positions = [x.split('\t')[-1].strip().split(',')[0][1:]
                         for x in sentence.split('\n') if len(x.split('\t')) > 1]

            if len(words) < options.max_length:
                for i in range(len(words), options.max_length):
                    words.append("<PAD>")
                    labels.append('O')
                    positions.append('0')
            else:
                if len([x for x in labels if x != 'O']):
                    print(labels)
                words = words[:options.max_length]
                labels = labels[:options.max_length]
                positions = positions[:options.max_length]
                
            
            assert len(words) == options.max_length
            
            for idx, trigger_position in enumerate(positions):
                if trigger_position != '0':
                    positions[idx] = idx
                else:
                    positions[idx] = -1

            # Generating all the relative distances for all the instances positive + negative
            all_positions = []
            for idx, trigger_position in enumerate(positions):
                if trigger_position > -1:
                    pos_trigger = list(range(-trigger_position, 0)) + \
                        list(range(0, len(words) - trigger_position))
#                    for i in range(len([x for x in words if '<PAD>' not in x])):
                    all_positions.append(pos_trigger)
                else:
                    
                    pos_trigger = list(range(-idx, 0)) + list(range(0, len(words) - idx))
                    all_positions.append(pos_trigger)
            
#            assert len(all_positions) == len(words) # Generate positions vectors for all the words in the sentence
            
            for positions_ in all_positions:
    
                window_words, window_labels, window_positions = words, labels, positions_
                
                if window_words[window_positions.index(0)] in punctuation:
#                    print('Skipping', window_words[window_positions.index(0)])
                    continue
                if window_words[window_positions.index(0)] == "<PAD>":
                    continue
                
                entry = {}
                entry['words'] = window_words

                entry['label'] = window_labels[window_positions.index(0)]
                entry['anchor_index'] = window_positions.index(0)
                                
                if max(window_positions) > MAX_DIST:
                    MAX_DIST = max(window_positions)
                if min(window_positions) < MIN_DIST:
                    MIN_DIST = min(window_positions)
                    
                entry['length'] = len(window_words)
                entry['mask'] = [1 if x != '<PAD>' else 0 for x in window_words]
                entry['dist'] = []
    
                for word_position, item in enumerate(
                        zip(window_words, window_labels, window_positions)):
                    word, label, distance = item
                    if word not in all_words:
                        all_words.append(word)
                    if word == '<PAD>':
                        entry['dist'].append(0)
                    else:
                        entry['dist'].append(distance)

                assert len(entry['words']) == options.max_length

                # Relative distances can only be positive numbers
                entry['dist'] = [options.max_length - 1 + x for x in window_positions] 
                
                word_data.append(entry)
                
                if count < 20:
                    print('Trigger: {}; Label: {}'.format(window_words[window_positions.index(0)], entry['label']))
                    print([word for word in entry['words'] if word != "<PAD>"])
                    print([distance for idx_distance, distance in enumerate(window_positions) if entry['words'][idx_distance] != "<PAD>"])
#                    print(entry['dist'])
                    print('*'*20)
                count += 1

        print('MAX_DIST', MAX_DIST)
        print('MIN_DIST', MIN_DIST)
        dico, word_to_id, id_to_word = word_mapping(all_words)

        for idx, entry in enumerate(word_data):
            word_data[idx]['indices'] = [word_to_id[word] for word in entry['words']]
            
            assert len(word_data[idx]['indices']) == len(word_data[idx]['dist'])
                
        if 'random' in options.embedding:
            word_embeds = np.random.normal(0.0, 0.5, (len(word_to_id), word_dim))
            word_embeds[0] = np.zeros(word_dim) #<PAD>
            word_embeds[1] = np.zeros(word_dim) #<UNK>
          
            with open('data/word_embeds_' + str(options.embedding) + '.pkl', 'wb') as f:
                pickle.dump(word_embeds, f)

        else:
            embeddings_model = load_embeddings(options.embedding)

            word_embeds = []
            print('Embedding encoding:')
            for word, _ in tqdm(word_to_id.items(), total=len(word_to_id.items())):
                if word in embeddings_model:
                    word_embeds.append(embeddings_model[word])
                elif word.lower() in embeddings_model:
                    word_embeds.append(embeddings_model[word.lower()])
                else:
                    try:
                        size_embeddings = embeddings_model.shape[1]
                    except:
                        size_embeddings = embeddings_model.vector_size
#                    word_embeds.append(np.random.normal(0.0, 0.5, size_embeddings))
#                    print('Unk:', word)
                    unknown_words += 1
                    word_embeds.append(np.zeros(size_embeddings))

            word_embeds = np.array(word_embeds)
            with open('data/word_embeds_' + str(options.embedding) + '.pkl', 'wb') as f:
                pickle.dump(word_embeds, f)

        print('Unknown words: ' + str((unknown_words * 100.0) / len(word_to_id)) + '%')

        with open('data/word_data_' + str(options.embedding) + '.pkl', 'wb') as f:
            pickle.dump(word_data, f)

    data = [x for x in word_data if x['label'] != 'O']
    other = [x for x in word_data if x['label'] == 'O']

    def check_test_type(test_types, label):
        #        import pdb;pdb.set_trace()
        for _type in test_types:
            _type = _type.replace('-', '_')
            if label in TYPES[_type]:
                return True
        return False

    train = [x for x in data if not check_test_type(test_type, x['label'])]
    rest = [x for x in data if check_test_type(test_type, x['label'])]

    valid = []
    test = []

    # For train
    for label, idx in label2index.items():
        samples = [x for x in train if x['label'] == label]
        if len(samples) > 0:
            for x in range(30 // (len(samples))):
                train += samples

    # ----------------------
    counter = collections.Counter()
    counter.update([x['label'] for x in train])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]

    print(
        'Accepted target classes with more than 20 examples for train:',
        accepted_target_classes)
    print(counter, '\n')
    # ----------------------

    # For dev and test
    counter = collections.Counter()
    counter.update([x['label'] for x in rest])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]

    print(
        'Accepted target classes with more than 20 examples for test and dev:',
        accepted_target_classes)

    print(counter, '\n')

    for t in accepted_target_classes:
        samples = [x for x in rest if x['label'] == t]
        valid += samples[:len(samples) // 2]
        test += samples[len(samples) // 2:]
    
#    print('Test length:', len(test))
#    new_test = []
#    for i in test:
#       if i not in new_test:
#          new_test.append(i)
#    test = new_test
#    print('Test length:', len(test))
    # ----------------------
    
    for examples in [('valid', valid), ('test', test)]:
        per_counter = collections.Counter()
        per_counter.update([x['label'] for x in examples[1]])
        per_accepted_target_classes = [
            k for k, v in per_counter.items() if v > 20]
        print('Accepted target classes with more than 20 examples for:',
              examples[0], per_accepted_target_classes)
        print(per_counter, '\n')
        # ----------------------

    # For other
    l = len(other) // 3
    train_other = other[:l]
    valid_other = other[l:2 * l]
    test_other = other[2 * l:]

    return train, valid, test, train_other, valid_other, test_other, word_embeds


def load_tac_dataset(options):
    import utils
    test_type = options.test_type
    data, label2idx = utils.read_tac_from_pickle()

    other = data['other']
    train = data['train']
    test = data['test']

    _data = train + test

    # Filter test from train:
    train = [x for x in _data if not x['label'].startswith(test_type)]
    rest = [x for x in _data if x['label'].startswith(test_type)]
    valid = []
    test = []

    for label, idx in label2idx.items():
        samples = [x for x in train if x['target'] == idx]
        if len(samples) > 0:
            for x in range(30 // (len(samples))):
                train += samples

    counter = collections.Counter()
    counter.update([x['target'] for x in rest])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]
    # print(accepted_target_classes)

    for t in accepted_target_classes:
        samples = [x for x in rest if x['target'] == t]
        valid += samples[:len(samples) // 2]
        test += samples[len(samples) // 2:]
    utils.print_label_distribution(train, valid, test)

    l = len(other) // 3
    train_other = other[:l]
    valid_other = other[l:2 * l]
    test_other = other[2 * l:]

    return train, valid, test, train_other, valid_other, test_other


class Fewshot(object):

    def __init__(
            self,
            positive_data,
            negative_data,
            features=DEFAULT_FEATURES,
            N=5,
            K=5,
            Q=4,
            O=0,
            noise=0.0,
            max_length=31):
        self.features = features
        self.positive_length = len(positive_data)
        self.negative_length = len(negative_data)
        self.max_length = max_length
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.noise = noise

        self.N = N
        self.K = K
        self.Q = Q
        self.O = O
        self.event2indices = {
            'O': [
                x for x in range(
                    self.negative_length)]}
        self.positive_class = {x['label'] for x in positive_data}
        for t in self.positive_class:
            indices = [idx for idx, x in enumerate(
                positive_data) if x['label'] == t]
            self.event2indices[t] = indices
        print('Positive_data: ', len(positive_data))
        print('negative_data: ', len(negative_data))
#        print(self.event2indices)

    def __len__(self):
        return 10000000

    def pack(self, items):
        data = {}
        for k in self.features:
            data[k] = [x[k] for x in items]
        return data

    def get_positive(self, scope):
        """

        :param scope: indices of sample of the same class
        :return:
        """
        indices = random.sample(scope, self.K + self.Q)
        # print(indices)

        items = [self.positive_data[i] for i in indices]
        support = self.pack(items[:self.K])
        query = self.pack(items[self.K:])

        # print(query['dist'])
        # exit(0)
        return support, query

    def get_negative(self):
        O, K = self.O, self.K
        scope = self.event2indices['O']
        indices = random.sample(scope, O * K)

        data = []
        for i in range(O):
            _indices = indices[i * K:(i + 1) * K]
            items = [self.negative_data[j] for j in _indices]
            _data = self.pack(items)
            data.append(_data)

        return data

    def __getitem__(self, idx):
        N, K, Q = self.N, self.K, self.Q
#        print('N samples:', self.N, 'from', len(self.positive_class))

        target_classes = random.sample(self.positive_class, N)
#        noise_classes = []
#        for class_name in self.event2indices.keys():
#            if not (class_name in target_classes):
#                noise_classes.append(class_name)

        support_set = []
        query_set = []
        query_label = []

        for i, class_name in enumerate(target_classes):  # N way
            scope = self.event2indices[class_name]
            support, query = self.get_positive(scope)
            support_set.append(support)
            query_set.append(query)

            query_label.append([i] * Q)

        other_set = self.get_negative()

        return self.pack(support_set), self.pack(
            query_set), self.pack(other_set), query_label


FEATURE_TYPES = {
    'target': torch.LongTensor,
    'indices': torch.LongTensor,
    'dist': torch.LongTensor,
    'length': torch.LongTensor,
    'mask': torch.FloatTensor,
    'anchor_index': torch.LongTensor,
    'adj_arc_in': torch.LongTensor,
    'adj_arc_out': torch.LongTensor,
    'adj_lab_in': torch.LongTensor,
    'adj_lab_out': torch.LongTensor,
    'adj_mask_in': torch.FloatTensor,
    'adj_mask_out': torch.FloatTensor,
    'adj_mask_loop': torch.FloatTensor,
    'adj_rel_fet': torch.FloatTensor,
    'mner': torch.LongTensor

}


def fewshot_negative_fn(items):
    feature_names = items[0][0].keys()
    positive = {k: [] for k in feature_names}
    query = {k: [] for k in feature_names}
    negative = {k: [] for k in feature_names}
    label = []
    for s, q, o, l in items:
        for k in feature_names:
            positive[k].append(s[k])
            query[k].append(q[k])
            negative[k].append(o[k])
            # print(len(s[k]), len(q[k]), len(o[k]))
        label.append(l)

    positive_ts = {k: FEATURE_TYPES[k](positive[k]) for k in feature_names}
    query_ts = {k: FEATURE_TYPES[k](query[k]) for k in feature_names}
    negative_ts = {k: FEATURE_TYPES[k](negative[k]) for k in feature_names}

    label_ts = torch.LongTensor(label)

    return positive_ts, query_ts, negative_ts, label_ts
