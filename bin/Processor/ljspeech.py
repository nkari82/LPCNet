# -*- coding: utf-8 -*-
# This code is copy and modify from https://github.com/keithito/tacotron.
"""Perform preprocessing and raw feature extraction."""

import os
import re
import numpy as np

from tensorflow_tts.utils import cleaners

valid_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

_pad = "_"
_eos = "~"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
symbols = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + [_eos]
)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


class LJSpeechProcessor(object):
    """LJSpeech processor."""
    def __init__(self, data_dir, cleaner_names, metadata_filename="metadata.csv"):
        self._data_dir = data_dir
        self._cleaner_names = cleaner_names
        self._max_ids_length = 0
        self._max_feat_size = 0
        self._speaker_name = "ljspeech"
        if data_dir:
            with open(os.path.join(data_dir, metadata_filename), encoding="utf-8") as f:
                self.items = [self._split_line(data_dir, line, "|") for line in f]

    def _split_line(self, data_dir, line, split):
        feat_file, text = line.strip().split(split)
        #print("feat: {}, text: {}".format(feat_file, text))
        feat_path = os.path.join(data_dir, "feats", f"{feat_file}.f32")
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)
        self._max_feat_size = max(self._max_feat_size, os.stat(feat_path).st_size)
        self._max_ids_length = max(self._max_ids_length, text_ids.shape[0])
        return text_ids, feat_path, self._speaker_name
        
    def max_ids_length(self):
        return self._max_ids_length
        
    def max_feat_size(self):
        return self._max_feat_size
        
    def text_to_sequence(self, text):
        global _symbol_to_id

        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += _symbols_to_sequence(_clean_text(text, [self._cleaner_names]))
                break
            sequence += _symbols_to_sequence(_clean_text(m.group(1), [self._cleaner_names]))
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)
        return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
