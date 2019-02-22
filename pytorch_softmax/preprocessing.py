import numpy as np

class Preprocess():

    def __init__(self):
        self.data_dir = '../data/extracted_emoji_sequences.txt'
        self.window_size = 8
        self.word2idx = None
        self.idx2word = None
        self.vocabulary_size = None
        self.idx_pairs = []
        self.buildingWordPairs()
    
    def tokenize_corpus(self, corpus):
        tokens = [x.split() for x in corpus]
        return tokens

    def onlyEmojiSequences(self, tokens):
        threshold_emojis = [x for x in tokens if len(x) > 1]
        return threshold_emojis

    def buildingWordPairs(self):
        corpus = open(self.data_dir).read().splitlines()

        tokenized_corpus = self.tokenize_corpus(corpus)
        emojiSequences = self.onlyEmojiSequences(tokenized_corpus)

        vocabulary = []
        for sentence in tokenized_corpus:
            for token in sentence:
                if token not in vocabulary:
                    vocabulary.append(token)

        self.word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        self.idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
        self.vocabulary_size = len(vocabulary)

        # for each sentence
        for sentence in tokenized_corpus:
            indices = [self.word2idx[word] for word in sentence]
            # for each word, threated as center word
            for center_word_pos in range(len(indices)):
                # for each window position
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_pos = center_word_pos + w
                    # make soure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    self.idx_pairs.append((indices[center_word_pos], context_word_idx))

        self.idx_pairs = np.array(self.idx_pairs)
