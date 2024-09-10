import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from omni_speech.constants import IGNORE_INDEX


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def _uniform_assignment(src_lens, tgt_lens):
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t


class SpeechGeneratorCTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers, n_dims, n_heads, n_inter_dims = list(map(int, config.ctc_decoder_config[1:-1].split(",")))
        _config = copy.deepcopy(config)
        _config.hidden_size = n_dims
        _config.num_hidden_layers = n_layers
        _config.num_attention_heads = n_heads
        _config.num_key_value_heads = n_heads
        _config.intermediate_size = n_inter_dims
        _config._attn_implementation = "flash_attention_2"
        self.upsample_factor = config.ctc_upsample_factor
        self.input_proj = nn.Linear(config.hidden_size, n_dims)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(_config, layer_idx) for layer_idx in range(n_layers)]
        )
        self.unit_vocab_size = config.unit_vocab_size
        self.output_proj = nn.Linear(n_dims, config.unit_vocab_size + 1)

    def upsample(self, reps, tgt_units=None):
        src_lens = torch.LongTensor([len(rep) for rep in reps]).to(reps[0].device)
        up_lens = src_lens * self.upsample_factor
        if tgt_units is not None:
            tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
            up_lens = torch.max(up_lens, tgt_lens)
        reps = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True)
        padding_mask = lengths_to_padding_mask(up_lens)
        mapped_inputs = _uniform_assignment(src_lens, up_lens).masked_fill(
            padding_mask, 0
        )
        copied_reps = torch.gather(
            reps,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), reps.size(-1)
            ),
        )
        copied_reps = copied_reps.masked_fill(padding_mask.unsqueeze(-1), 0)
        position_ids = torch.arange(0, max(up_lens)).unsqueeze(0).expand(len(reps), -1).to(device=copied_reps.device)
        return copied_reps, ~padding_mask, position_ids
    
    def forward(self, tgt_reps, labels, tgt_units):
        tgt_label_reps = []
        for tgt_rep, label in zip(tgt_reps, labels):
            tgt_label_reps.append(tgt_rep[label != IGNORE_INDEX])
        hidden_states, attention_mask, position_ids = self.upsample(tgt_label_reps, tgt_units)
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
        ctc_logits = self.output_proj(hidden_states)
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_lens = attention_mask.long().sum(dim=-1)
        ctc_tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)
        ctc_tgt_flat = tgt_units.masked_select(ctc_tgt_mask)
        ctc_loss = F.ctc_loss(
            ctc_lprobs.transpose(0, 1),
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            reduction="sum",
            zero_infinity=True,
            blank=self.unit_vocab_size
        )
        ctc_loss /= ctc_tgt_lens.sum().item()
        return ctc_loss
    
    def predict(self, tgt_reps):
        hidden_states, attention_mask, position_ids = self.upsample([tgt_reps])
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
        ctc_logits = self.output_proj(hidden_states)
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_pred = ctc_lprobs.argmax(dim=-1).masked_fill_(~attention_mask, self.unit_vocab_size)
        return ctc_pred