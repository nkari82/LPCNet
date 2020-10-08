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

    class Generator(object):
        def __init__(self):
            self._max_seq_length = 0
            self._max_feat_size = 0
    
        def __call__(self, rootdir, tid, seq, speaker):
            feat_path = os.path.join(rootdir, "feats", f"{tid}.f32")
            self._max_feat_size = max(self._max_feat_size, os.stat(feat_path).st_size)
            self._max_seq_length = max(self._max_seq_length, seq.shape[0])
            
            return tid, seq, feat_path, speaker
            
        def max_seq_length(self):
            return self._max_seq_length
        
        def max_feat_length(self):
            return self._max_feat_size // 4 # // float32 4byte
            
        def complete(self):
            pass
    
    def __init__(self, rootdir, **kwargs):  
        self._tagger = MeCab.Tagger('')
        self._symbol_to_id = {c: i for i, c in enumerate(symbols)}
        self._id_to_symbol = {i: c for i, c in enumerate(symbols)}
        self._rootdir = rootdir
        self._speaker = "tsuchiya"
        self._metadata = kwargs.get('metadata',"metadata.csv")
        self._generator = kwargs.get('generator', self.Generator())
    
        self.items = []
        if rootdir:
            with open(os.path.join(rootdir, self._metadata), encoding="utf-8") as f:
                for line in f:
                    item = self._parse(line, "|")
                    item if item is None else self.items.append(item)
                    
            self._generator.complete()

    def _parse(self, line, split):
        tid, text = line.strip().split(split)
        item = None
        try:
            seq = np.asarray(self.text_to_sequence(text), np.int32)
            item = self._generator(self._rootdir, tid, seq, self._speaker)
        except Exception as ex:
            print("tid: {}, err: {}, text: {}".format(tid, ex, text))
        return item
        
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
        return self._generator.max_seq_length();
        
    def max_feat_length(self):
        return self._generator.max_feat_length();
        
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
        
#processor = JSpeechProcessor(rootdir=None)
#print(processor.vocab_size())  167
#processor.text_to_sequence('また東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い。')
#[71, 40, 49, 101, 33, 55, 81, 101, 52, 29, 41, 13, 72, 80, 101, 19, 101, 49, 81, 57, 85, 84, 32, 78, 81, 101, 51, 72, 80, 101, 19, 101, 55, 42, 78, 101, 19, 101, 52, 56, 13, 30, 85, 84, 28, 49, 75, 19, 101, 13, 4, 166]