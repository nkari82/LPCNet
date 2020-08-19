import jaconv

class JSpeechProcessor(object):
    def __init__(self, data_dir, cleaner_names, metadata_filename="metadata.csv"):
        self._data_dir = data_dir
        self._cleaner_names = cleaner_names
        self._max_ids_length = 0
        self._max_feat_size = 0
        self._eos = 0
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
        
    def _add_punctuation(text):
        last = text[-1]
        if last not in [".", ",", "、", "。", "！", "？", "!", "?"]:
            text = text + "。"
        return text

    def _normalize_delimitor(text):
        text = text.replace(",", "、")
        text = text.replace(".", "。")
        text = text.replace("，", "、")
        text = text.replace("．", "。")
        return text

    def max_ids_length(self):
        return self._max_ids_length
        
    def max_feat_size(self):
        return self._max_feat_size
        
    def vocab_size(self):
        return 0xffff
        
    def text_to_sequence(text):
        # cleaner
        for c in [" ", "　", "「", "」", "『", "』", "・", "【", "】","（", "）", "(", ")"]:
            text = text.replace(c, "")
        text = text.replace("!", "！")
        text = text.replace("?", "？")

        text = self._normalize_delimitor(text)
        text = jaconv.normalize(text)
        text = jaconv.hira2kata(text)
        text = self._add_punctuation(text)

        return [ord(c) for c in text] + [self._eos]  # EOS

    def sequence_to_text(seq):
        return "".join(chr(n) for n in seq)