
import stanza
from nltk.tree import Tree
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


def get_all_nps(tree, full_sent=None):
    start = 0
    end = len(tree.leaves())

    def single_NP(tree):

        num_sub_nodes = len(tree)

        for i in range(num_sub_nodes):
            if tree[i].label() == "NP":
                return False

        return True

    def get_sub_nps(tree, left, right):
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []
        sub_nps = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        # if tree.label() == 'NP' and n_leaves > 1 and single_NP(tree):
        if tree.label() == 'NP' and n_leaves > 1 :
            sub_nps.append([" ".join(tree.leaves()), (int(left), int(right))])
        for i, subtree in enumerate(tree):
            sub_nps += get_sub_nps(subtree, left=left + offset[i], right=left + offset[i] + n_subtree_leaves[i])
        return sub_nps

    all_nps = get_sub_nps(tree, left=start, right=end)
    lowest_nps = []
    for i in range(len(all_nps)):
        span = all_nps[i][1]
        lowest = True
        for j in range(len(all_nps)):
            span2 = all_nps[j][1]
            if span2[0] >= span[0] and span2[1] <= span[1]:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_nps[i])

    all_nps, spans = map(list, zip(*all_nps))
    if full_sent and full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [(start, end)] + spans

    return all_nps, spans, lowest_nps


class ObjectSplit:

    def __init__(self, prompt):
        super().__init__()

        self.nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, constituency')
        self.prompt = prompt

    def __call__(self):
        # 1. preprocess prompt
        prompt = [self.prompt.lower().strip().strip(".").strip()]
        doc = self.nlp(prompt[0])

        # 2. use stanza to split nouns
        word_tree = Tree.fromstring(str(doc.sentences[0].constituency))
        print(word_tree.leaves())
        nps, spans, noun_chunk = get_all_nps(word_tree, prompt[0])
        print("nps:", nps)
        print("spans:", spans)
        print("noun_chunk:", noun_chunk)

        return dict(zip(nps, spans)), word_tree.leaves(), noun_chunk


class AlignEmbeds:

    def __init__(self, words, obj_info, tokenizer, text_encoder, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.text_encoder = text_encoder.to(device)
        self.nps = list(obj_info.keys())
        self.spans = list(obj_info.values())
        self.words = words

    def _encode_embedding(self, prompt: str):
        tokens = self.tokenizer(prompt,
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt").input_ids.to(self.device)
        # print(tokens)
        prompt_cond_embeds = self.text_encoder(tokens).last_hidden_state
        # print(prompt_cond_embeds.shape)
        # print(prompt_cond_embeds)

        return prompt_cond_embeds

    def _align_sequence(self, np_embeds, spans, pos_to_replace):

        main_seq_embed = np_embeds[0]  # shape [1,77,768]
        if len(spans) < 2:
            return main_seq_embed

        if pos_to_replace is not None:
            # 默认就为所有的单个名词词组
            pos_to_replace = [i for i in range(1, len(spans))]

        for i in pos_to_replace:
            replaced_np_embed = np_embeds[i]

            span = spans[i]
            start, end = span[0] + 1, span[1] + 1  # shift
            seq_length = end - start

            main_seq_embed[:, start:end, :] = replaced_np_embed[:, 1:seq_length + 1, :]

            print(f"Replaced embedding: {self.nps[i]}")

        return main_seq_embed

    def __call__(self, pos_to_replace=None):
        # pos_to_replace : 代表第几个 object 进行替换
        # 1. tokenizer and embedding
        
        if pos_to_replace is not None:
            spans = [ ]
            nps = []
            for span in pos_to_replace:
                spans.append(span)
                prompt_to_embed = " ".join(self.words[span[0]:span[1]])
                nps.append(prompt_to_embed)
            
            self.nps = nps
            self.spans = spans
            print("update nps:", nps)
            print("update spans:", spans)
        
        np_embeds = [self._encode_embedding(np) for np in self.nps]

        # 2. replace embedding
        if len(np_embeds) != 1:
            cond_embeddings = self._align_sequence(np_embeds, self.spans, pos_to_replace)
        else:
            cond_embeddings = np_embeds[0]

        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        prompt_embeds = torch.cat([uncond_embeddings, cond_embeddings])

        return prompt_embeds

