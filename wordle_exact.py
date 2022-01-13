import logging
import numpy as np
import urllib.request
from tqdm import tqdm
from collections import Counter


def read_file(file_path):
	"""
	Read a file from disk/ download from a web URL
	"""
	if file_path.startswith('https://') or file_path.startswith('http://'):
		with urllib.request.urlopen(file_path) as f:
			lines = f.read().decode('utf-8').split('\n')
	else:
		with open(file_path) as f:
			lines = f.readlines()
	return lines


class Wordle():
	"""
	Create an instance of a Wordle game to maintain the state, also suggest the next best guesses
	Args:
		- _len: length of the wordle game
		- words_data_files: dict of word list types to file paths
		- onlyalpha: Only consider words from the seed word list which are purely alphabetic
	"""
	def __init__(self, _len, words_data_files, onlyalpha=True):
		self._len = _len
		self.hints = []
		self.read_initial_words(words_data_files, onlyalpha)

	def read_words(file_path, onlyalpha, _len):
		"""
		Read words from a file with filters
		"""
		words = set()
		for line in read_file(file_path):
			word = line.strip()
			if len(word) != _len:
				continue
			if onlyalpha and (not word.isalpha()):
				continue
			words.add(word)
		return np.array(list(words))

	def get_label(self, word, target):
		"""
		Get wordle label using encoding
			0: character not present
			1: character present
			2: character at correct location
		"""
		ans = ['0' for i in range(len(word))]
		char_counts = dict(Counter(list(target)))
		for pos, char in enumerate(word):
			if target[pos] == char:
				ans[pos] = '2'
				char_counts[char] -= 1
		for pos, char in enumerate(word):
			if ans[pos] != '2' and char_counts.get(char, 0) > 0:
				ans[pos] = '1'
				char_counts[char] -= 1
		return ''.join(ans)

	def read_initial_words(self, words_data_files, onlyalpha):
		print('Reading initial words')
		self.words = {}
		for word_type, fname in words_data_files.items():
			words = read_words(fname, onlyalpha, self._len)
			self.words[word_type] = [words, words]  # original list, pruned list
			logger.info(f'Number of {word_type} words {len(words)}')

	def add_word_hint(self, word, label):
		"""
		Store revealed wordle hints
		"""
		self.hints.append((word, label))
		for word_type, words in self.words.items():
			logger.info(f'Pruning {word_type} words')
			words[1] = self.prune_words(words[1])

	def prune_words(self, words):
		"""
		Prune argument list of words to keep only those words which satisfy all the hints so far
		"""
		pruned_words = [word for word in words \
						if all(self.get_label(hword, word) == hlabel for hword, hlabel in self.hints)]
		logger.info(f'Pruned from {len(words)} to {len(pruned_words)}')
		return pruned_words

	def get_score(self, word, anstype):
		"""
		Score is proportional to expected length of pruned word list after word is played
		For a given word, answer words can be partitioned based on the response when the word is played
		Any response will prune the word list to the partition of the correct word
		Hence, expected length of pruned list is proportional to sum(square(size of partition))
		"""
		return (np.array(list(Counter([self.get_label(word, w) for w in self.words[anstype][1]]).values()))**2).sum()

	def predict(self, K=1, searchtype='allowed', anstype='answers', pruned=True):
		"""
		Args:
			- K = Number of results to return
			- searchtype = key of words list to search on. Recommended to be extended list until number of words in final candidate 
			set is 1-2 and the answer is obvious
			- anstype = key of words list which should be used as answers
			- pruned = If False, older hints can be discarded in the next guess. Allows to select more informative words at the
			expense of correctness
		"""
		words = self.words[searchtype][pruned]
		scores = [self.get_score(word, anstype) for word in tqdm(words)]
		for idx in np.argsort(scores)[:K]:
			print(f'WORD: {words[idx]}\tSCORE: {scores[idx]}')


if __name__ == "__main__":
	words_data_files = {
		'allowed': 'combined_wordlist.txt',  # 'https://raw.githubusercontent.com/Kinkelin/WordleCompetition/main/data/official/combined_wordlist.txt'
		'answers': 'shuffled_real_wordles.txt'  # 'https://raw.githubusercontent.com/Kinkelin/WordleCompetition/main/data/official/shuffled_real_wordles.txt'
	}
	logger = logging.getLogger(__name__)
	logging.basicConfig(
	        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	        datefmt="%m/%d/%Y %H:%M:%S",
	        level=logging.INFO,
	)
	game = Wordle(5, words_data_files, onlyalpha=False)
	game.add_word_hint('roate', '00101')
	import pdb; pdb.set_trace()
	# game.predict(K=10, searchtype='allowed', pruned=False)
	# game.add_word_hint('roate', '00000')
