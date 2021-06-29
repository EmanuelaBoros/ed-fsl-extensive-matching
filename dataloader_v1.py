from itertools import islice
import collections
import random
import numpy as np
import torch
from tqdm import tqdm

def merge(d1, d2):
    for id, sample in d1.items():
        sample.update(d2[id])
    return d1


def get_labels():
    return {
        'O': 0,
        'Business': 1,
        'Conflict': 2,
        'Contact': 3,
        'Life': 4,
        'Movement': 5,
        'Justice': 6,
        'Transaction': 7,
        'Personnel': 8
    }


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
    for items in item_list:
        for item in items:
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


def load_ace_dataset(options, path):
    #    import utils
    test_type = options.test_type
    import embeddings as emb

    half_window = int(options.max_length / 2)
    DISTANCE_MAPPING = {}
    minDistance = -half_window
    maxDistance = half_window

    DISTANCE_MAPPING['<PAD>'] = 0
    for dis in range(minDistance, maxDistance + 1):
        DISTANCE_MAPPING[dis] = len(DISTANCE_MAPPING)

    print('test_type', test_type)
#    embeddings_model = emb.fetch_GloVe()

#    word_data = utils.read_pickle('files/{}/word.proc'.format(options.dataset))
    # utils.read_pickle('files/{}/label2index.proc'.format(options.dataset))
    label2index = get_labels()

#    if options.encoder == 'gcn':
#        matrix_data = utils.read_pickle('files/{}/matrix.proc'.format(options.dataset))
#        word_data = merge(matrix_data, word_data)

#    print(word_data['nw/timex2norm/AFP_ENG_20030327.0022-29'].keys())

    with open(path, 'r') as f:
        sentences = f.read().split('\n\n')

    word_dim = 300
    word_embeds = np.random.uniform(-np.sqrt(0.06),
                                    np.sqrt(0.06), (100000, word_dim))

    def get_event_type(label):
        if len(label) == 1:
            return label
        for key in TYPES.keys():
            subtypes = TYPES[key]
            for subtype in subtypes:
                if label[2:] in subtype.replace('-', '_'):
                    return key
        return label

    all_words = []
    word_data = []
    unknown_words = 0
    count = 0
    for sentence in tqdm(sentences, total=len(sentences)):
        if 'DOCSTART' in sentence:
            continue
        words = [x.split('\t')[0].strip()
                 for x in sentence.split('\n') if len(x.split('\t')) > 1]
        labels = [get_event_type(x.split('\t')[1].strip())
                  for x in sentence.split('\n') if len(x.split('\t')) > 1]
        positions = [x.split('\t')[-1].strip().split(',')[0][1:]
                     for x in sentence.split('\n') if len(x.split('\t')) > 1]

#        print(labels)
        for i in range(half_window):
            words.append("<PAD>")
            words.insert(0, "<PAD>")
            labels.append("<PAD>")
            labels.insert(0, "<PAD>")
            positions.append("<PAD>")
            positions.insert(0, "<PAD>")
            
        
        for item in zip(sliding_window(words, options.max_length),
                        sliding_window(labels, options.max_length),
                        sliding_window(positions, options.max_length)):

            window_words, window_labels, window_positions = item

            entry = {}
            entry['words'] = window_words
            entry['label'] = window_labels[half_window]
            entry['indices'] = []
            entry['anchor_index'] = half_window
            entry['length'] = len(
                [x for x in window_words if '<PAD>' not in x])
            entry['mask'] = [1 if x != '<PAD>' else 0 for x in window_words]
            entry['dist'] = []

            distances = list(range(-half_window, half_window + 1))

            for word_position, item in enumerate(
                    zip(window_words, window_labels, window_positions, distances)):
                word, label, position, distance = item
                if word not in all_words:
                    all_words.append(word)
                entry['indices'].append(all_words.index(word))

                entry['dist'].append(DISTANCE_MAPPING[distance])

            word_data.append(entry)
            assert len(entry['indices']) == len(entry['dist'])
            if count < 5:
                print(entry)
            count += 1
#            import pdb;pdb.set_trace()

#                if len(label)> 1:
#                    print(label, position)
#                for feature in [
#                    'indices',
#                    'dist',
#                    'length',
#                    'mask',
#                        'anchor_index']:
            #                if word in embeddings_model:
            #                    entry['indices'] = embeddings_model[word]
            #                    all_words.append(word)
            #                elif word.lower() in embeddings_model:
            #                    entry['indices'] = embeddings_model[word.lower()]
            #                    all_words.append(word.lower())
            #                else:
            #                    unknown_words += 1

    dico, word_to_id, id_to_word = word_mapping(all_words)
    print('Uknown words: ' + str((unknown_words * 100.0) / len(word_to_id)) + '%')

#    import pdb;pdb.set_trace()

#    data = [x for idx, x in word_data.items() if x['label'] != 'O']
#    other = [x for idx, x in word_data.items() if x['label'] == 'O']

    data = [x for x in word_data if x['label'] != 'O']
    other = [x for x in word_data if x['label'] == 'O']

#    # Filter test from train:
#    train = [x for x in data if not x['label'].startswith(test_type)]
#    rest = [x for x in data if x['label'].startswith(test_type)]
    def check_test_type(test_types, label):
        #        import pdb;pdb.set_trace()
        for _type in test_types:
            _type = _type.replace('-', '_')
            if _type in label:
                return True
        return False

    train = [x for x in data if not check_test_type(test_type, x['label'])]
    rest = [x for x in data if check_test_type(test_type, x['label'])]

#    import pdb;pdb.set_trace()
    valid = []
    test = []

    # For train
    for label, idx in label2index.items():
        samples = [x for x in train if x['label'] == label]
        if len(samples) > 0:
            for x in range(30 // (len(samples))):
                train += samples

    # For dev and test
    counter = collections.Counter()
    counter.update([x['label'] for x in rest])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]

    for t in accepted_target_classes:
        samples = [x for x in rest if x['label'] == t]
        valid += samples[:len(samples) // 2]
        test += samples[len(samples) // 2:]

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
            noise=0.0):
        self.features = features
        self.positive_length = len(positive_data)
        self.negative_length = len(negative_data)
        self.max_length = 31
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.noise = noise

        self.N = N
        self.K = K
        self.Q = Q
        self.O = O
        self.event2indices = {
            'other': [
                x for x in range(
                    self.negative_length)]}
        self.positive_class = {x['label'] for x in positive_data}
        for t in self.positive_class:
            indices = [idx for idx, x in enumerate(
                positive_data) if x['label'] == t]
            self.event2indices[t] = indices
        print('Positive_data: ', len(positive_data))
        print('negative_data: ', len(negative_data))

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
        scope = self.event2indices['other']
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
        print('N samples:', self.N)
        print(self.positive_class)
        target_classes = random.sample(self.positive_class, N)
        noise_classes = []
        for class_name in self.event2indices.keys():
            if not (class_name in target_classes):
                noise_classes.append(class_name)

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
