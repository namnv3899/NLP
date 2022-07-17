import torch
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE

from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel
import numpy as np


class BPE:
    def __init__(self, path) -> None:
        self.bpe_codes = path


class EmbedingWithRoberta:
    def __init__(self, config_path, model_path, dict_path, bpe_path) -> None:
        # Load model
        self.config = RobertaConfig.from_pretrained(config_path)
        self.phobert = RobertaModel.from_pretrained(model_path, config=self.config).to(
            torch.device("cuda:0")
        )
        self.load_bpe(bpe_path)
        self.load_dict(dict_path)

    def load_bpe(self, bpe_path):
        # Load BPE encoder
        self.bpe = fastBPE(BPE(bpe_path))

    def load_dict(self, dict_path):
        # Load the dictionary
        self.vocab = Dictionary()
        self.vocab.add_from_file(dict_path)

    def text2ids(self, text):
        # INPUT TEXT IS WORD-SEGMENTED!
        # Encode the line using fastBPE & Add prefix <s> and suffix </s>
        subwords = "<s> " + self.bpe.encode(text) + " </s>"

        # Map subword tokens to corresponding indices in the dictionary
        input_ids = (
            self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False)
            .long()
            .tolist()
        )

        return input_ids

    def get_emb_vector(self, input_ids):
        input_ids = torch.tensor([input_ids]).to(torch.long)

        with torch.no_grad():
            features = self.phobert(input_ids.to(torch.device("cuda:0")))
        emb_vecs = features[0].cpu().numpy()[0]  # [1:-1]

        return emb_vecs


class Embeding:
    def __init__(self, phobert_data) -> None:
        self.load_bert(phobert_data)

    def load_bert(self, phobert_data):
        self.phobert = AutoModel.from_pretrained(
            phobert_data, local_files_only=True
        ).to(torch.device("cuda:0"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            phobert_data, local_files_only=True, use_fast=False
        )

    def text2ids(self, text):
        return self.tokenizer.encode(text)

    def get_emb_vector(self, input_ids):
        input_ids = torch.tensor(np.array([input_ids])).to(torch.long)
        with torch.no_grad():
            features = self.phobert(input_ids.to(torch.device("cuda:0")))

        emb_vecs = features[0].cpu().numpy()[0]  # [1:-1]

        return emb_vecs
