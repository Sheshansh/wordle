import logging
import numpy as np
from collections import Counter
import urllib.request


# ALLOWED_WORDS_FILE = 'https://raw.githubusercontent.com/Kinkelin/WordleCompetition/main/data/official/official_allowed_guesses.txt'
# ANSWERS_WORDS_FILE = 'https://raw.githubusercontent.com/Kinkelin/WordleCompetition/main/data/official/shuffled_real_wordles.txt'
ALLOWED_WORDS_FILE = 'combined_wordlist.txt'
ANSWERS_WORDS_FILE = 'shuffled_real_wordles.txt'

logger = logging.getLogger(__name__)


def get_information(event_counts):
	event_counts = np.array(event_counts).astype(np.float)
	# p*log(p) is 0 for p=0
	event_counts = event_counts[event_counts>0]
	if len(event_counts) == 0:
		return 0
	event_counts /= event_counts.sum()
	info = event_counts*np.log(event_counts)
	return -info.sum()


def read_file(file_path):
	if file_path.startswith('https://') or file_path.startswith('http://'):
		with urllib.request.urlopen(file_path) as f:
			lines = f.read().decode('utf-8').split('\n')
	else:
		with open(file_path) as f:
			lines = f.readlines()
	return lines


def read_words(file_path, onlyalpha, _len):
	words = set()
	for line in read_file(file_path):
		word = line.strip()
		if len(word) != _len:
			continue
		if onlyalpha and (not word.isalpha()):
			continue
		words.add(word)
	return words


class Wordle():
	"""
	Create an instance of a Wordle game to maintain the state, also suggest the next best guesses
	Arguments:
		_len: length of the wordle game
		onlyalpha: Only consider words from the seed word list which are purely alphabetic
	"""
	def __init__(self, _len, onlyalpha=True):
		self._len = _len
		self.onlyalpha = onlyalpha
		self.read_initial_words()
		self.update_chars()
		self.update_counts()

	def read_initial_words(self):
		print('Reading initial words')
		self.allowed_words = read_words(ALLOWED_WORDS_FILE, self.onlyalpha, self._len)
		self.answer_words = read_words(ANSWERS_WORDS_FILE, self.onlyalpha, self._len)
		self.num_allowed_words = len(self.allowed_words)
		self.num_answer_words = len(self.answer_words)
		logger.info(f'Number of allowed words {self.num_allowed_words}')
		logger.info(f'Number of answer words {self.num_answer_words}')

	def add_word_hint(self, word, label):
		"""
		Add information for a word where label is known and encoded via
			0: character not present
			1: character present
			2: character at correct location
		"""
		for char in label:
			assert char in ['0', '1', '2'], 'label should consist of 0, 1 and 2 only'
		assert len(word) == len(label), 'label and word should have equal length'
		for pos, (char, label) in enumerate(zip(word, label)):
			self.add_char_hint(char, label, pos)

	def add_char_hint(self, char, label, pos):
		words = set()
		for word in self.answer_words:
			charinword = (char in word)
			if label == '0':
				if charinword:
					continue
			if label == '1':
				if not charinword:
					continue
				if word[pos] == char:
					continue
			if label == '2':
				if not charinword:
					continue
				if word[pos] != char:
					continue
			words.add(word)
		self.answer_words = words
		self.num_answer_words = len(self.answer_words)
		logger.info(f'Number of words {self.num_answer_words}')
		self.update_chars()
		self.update_counts()

	def update_chars(self):
		self.chars = set()
		for word in self.answer_words:
			self.chars.update(list(word))
		logger.info(f'Number of characters {len(self.chars)}')

	def update_counts(self):
		self.char_pos_counts = {}
		self.char_in_counts = {}
		self.char_notin_counts = {}
		for char in self.chars:
			self.char_pos_counts[char] = {}
			self.char_in_counts[char] = 0
			self.char_notin_counts[char] = 0
			for pos in range(self._len):
				self.char_pos_counts[char][pos] = 0
		for word in self.answer_words:
			charset = set(list(word))
			for char in charset:
				self.char_in_counts[char] += 1
			for char in self.chars.difference(charset):
				self.char_notin_counts[char] += 1
			for pos, char in enumerate(word):
				self.char_pos_counts[char][pos] += 1

	def get_char_info(self, char):
		if char not in self.chars:
			return 0
		return get_information([self.char_in_counts[char], self.char_notin_counts[char]])

	def get_char_pos_info_approx(self, char, pos):
		if char not in self.chars:
			return 0
		event_counts = [self.char_pos_counts[char][pos],
						self.char_in_counts[char]-self.char_pos_counts[char][pos],
						self.char_notin_counts[char]]
		return get_information(event_counts)

	def get_word_info_approx(self, word):
		info = 0
		wordchars = set()
		for pos, char in enumerate(word):
			info += self.get_char_pos_info_approx(char, pos)
			# if char is not first occurence, delete the information we already had about word presence
			if char in wordchars:
				info -= self.get_char_info(char)
			else:
				wordchars.add(char)			
		return info

	def predict(self, K=1, can_ignore_hints=False):
		if can_ignore_hints:
			words = np.array(list(self.allowed_words))
		else:
			words = np.array(list(self.answer_words))
		scores = np.array([self.get_word_info_approx(word) for word in words])
		scores_order = np.argsort(scores)
		for idx in scores_order[::-1][:K]:
			print(f'WORD: {words[idx]}\tSCORE: {scores[idx]}')


if __name__ == "__main__":
	# set up logging
	logging.basicConfig(
	        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	        datefmt="%m/%d/%Y %H:%M:%S",
	        level=logging.INFO,
	)
	game = Wordle(5, onlyalpha=False)
	game.predict(K=10)
	import pdb; pdb.set_trace()
