# -*- coding: utf-8 -*-
import sys
import unicodedata
import MeCab
import os
import numpy as np

#http://pop365.cocolog-nifty.com/blog/2015/03/windows-64bit-m.html
#pip install mecab-python3
#pip install unidic-lite

_pad = "pad"
_eos = "eos"
_punctuation = [".",",","、","。","！","？","!","?"]
_cleaner = [" ","　","「","」","『","』","・","【","】","（","）","(", ")"]
_letters = [chr(_) for _ in range(0x30A0, 0x30FF)]  # katakana
_numbers = "0123456789"
_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

symbols = [_pad] + _punctuation + _letters + list(_alphabet) + list(_numbers) + [_eos]

class JSpeechProcessor(object):
    def __init__(self, data_dir, cleaner_names="", metadata_filename="metadata.csv"):  
        self._tagger = MeCab.Tagger('')
        self._symbol_to_id = {c: i for i, c in enumerate(symbols)}
        self._id_to_symbol = {i: c for i, c in enumerate(symbols)}
        self._data_dir = data_dir
        self._cleaner_names = cleaner_names
        self._max_seq_length = 0
        self._max_feat_size = 0
        self._speaker_name = "tsuchiya"
        
        if data_dir:
            with open(os.path.join(data_dir, metadata_filename), encoding="utf-8") as f:
                self.items = [self._split_line(data_dir, line, "|") for line in f]
    
    def _split_line(self, data_dir, line, split):
        tid, text = line.strip().split(split)
        feat_path = os.path.join(data_dir, "feats", f"{tid}.f32")
        text_seq = np.asarray(self.text_to_sequence(text), np.int32)
        self._max_feat_size = max(self._max_feat_size, os.stat(feat_path).st_size)
        self._max_seq_length = max(self._max_seq_length, text_seq.shape[0])
        return tid, text_seq, feat_path, self._speaker_name
        
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
        text = text.replace('〜', 'ー').replace('～', 'ー')
        text = text.replace("’", "'").replace('”', '"').replace('“', '``')
        text = text.replace('˗', '-').replace('֊', '-')
        text = text.replace('‐', '-').replace('‑', '-').replace('‒', '-').replace('–', '-')
        text = text.replace('⁃', '-').replace('⁻', '-').replace('₋', '-').replace('−', '-')
        text = text.replace('﹣', 'ー').replace('－', 'ー').replace('—', 'ー').replace('―', 'ー')
        text = text.replace('━', 'ー').replace('─', 'ー').replace(',', '、').replace('.', '。')
        text = text.replace('，', '、').replace('．', '。').replace('!', '！').replace('?', '？')
        return unicodedata.normalize('NFKC', text)

    def max_seq_length(self):
        return self._max_seq_length
        
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
        #print(sequence)
        return sequence

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
        #print(text)
        
#processor = JSpeechProcessor(data_dir=None)
#print(processor.vocab_size())  167
#processor.text_to_sequence('また東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い。')
#[71, 40, 49, 101, 33, 55, 81, 101, 52, 29, 41, 13, 72, 80, 101, 19, 101, 49, 81, 57, 85, 84, 32, 78, 81, 101, 51, 72, 80, 101, 19, 101, 55, 42, 78, 101, 19, 101, 52, 56, 13, 30, 85, 84, 28, 49, 75, 19, 101, 13, 4, 166]