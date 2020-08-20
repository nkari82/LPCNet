# -*- coding: utf-8 -*-
import jaconv
import MeCab
import sys

#http://pop365.cocolog-nifty.com/blog/2015/03/windows-64bit-m.html
#pip install mecab-python3
#pip install unidic-lite

_pad = "pad"
_eos = "eos"
_punctuation = [".", ",", "、", "。", "！", "？", "!", "?"]
_cleaner = [" ", "　", "「", "」", "『", "』", "・", "【", "】","（", "）", "(", ")"]
_letters = [chr(_) for _ in range(0x30A0, 0x30FF)]  # katakana
symbols = [_pad] + _punctuation + _letters + [_eos]

class JSpeechProcessor(object):
    def __init__(self, data_dir, cleaner_names, metadata_filename="metadata.csv"):
        self._tagger = MeCab.Tagger('')
        self._symbol_to_id = {c: i for i, c in enumerate(symbols)}
        self._id_to_symbol = {i: c for i, c in enumerate(symbols)}
        self._data_dir = data_dir
        self._cleaner_names = cleaner_names
        self._max_ids_length = 0
        self._max_feat_size = 0
        self._speaker_name = "tsuchiya"
        
        if data_dir:
            with open(os.path.join(data_dir, metadata_filename), encoding="utf-8") as f:
                self.items = [self._split_line(data_dir, line, "|") for line in f]
    
    def _split_line(self, data_dir, line, split):
        feat_file, text = line.strip().split(split)
        feat_path = os.path.join(data_dir, "feats", f"{feat_file}.f32")
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)
        self._max_feat_size = max(self._max_feat_size, os.stat(feat_path).st_size)
        self._max_ids_length = max(self._max_ids_length, text_ids.shape[0])
        return text_ids, feat_path, self._speaker_name
        
    def _pronunciation(self, text):
        result = self._tagger.parse(text)
        tokens=[]
        for line in result.split("\n")[:-1]:
            s = line.split("\t")
            if len(s) == 1:
                break
            tokens.append(s[0] if s[0] in _punctuation else s[1])
        return ''.join(token for token in tokens)
    
    def number_to_japanese(self, text):
        pass
        
    def _normalize(self, text):
        text = text.replace(",", "、")
        text = text.replace(".", "。")
        text = text.replace("，", "、")
        text = text.replace("．", "。")
        text = text.replace("!", "！")
        text = text.replace("?", "？")
        text = jaconv.normalize(text)
        return text

    def max_ids_length(self):
        return self._max_ids_length
        
    def max_feat_size(self):
        return self._max_feat_size
        
    def vocab_size(self):
        return len(symbols)
        
    def text_to_sequence(self, text):
        sequence = []
        text = self._clean_text(text)
        text = self._normalize(text)
        text = self._pronunciation(text)
        sequence = self._symbols_to_sequence(text)
        sequence += self._symbols_to_sequence([_eos])

    def sequence_to_text(seq):
        return "".join(chr(n) for n in seq)
        
    def _symbols_to_sequence(self, symbols):
        return [self._symbol_to_id[s] for s in symbols]
    
    def _clean_text(self, text):
        for c in _cleaner:
            text = text.replace(c, "")
        return text
        
    def text_to(self, text):
        text = self._clean_text(text)
        text = self._normalize(text)
        text = self._pronunciation(text)
        print(text)
        
#processor = JSpeechProcessor()
#processor.text_to_sequence('また東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い。')

