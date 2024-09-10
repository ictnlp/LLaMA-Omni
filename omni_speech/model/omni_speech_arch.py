# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod

import torch

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.utils import lengths_to_padding_mask


class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)

        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)

    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder

        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'))


class OmniSpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector

    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        if "whisper" in speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f'Unknown speech encoder: {speech_encoder}')
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features

    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        speech_features = self.encode_speech(speech, speech_lengths)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            if num_speech == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                continue

            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels