#!/bin/bash

ROOT=$1

VOCODER_CKPT=vocoder/g_00500000
VOCODER_CFG=vocoder/config.json

python omni_speech/infer/infer.py \
    --model-path Llama-3.1-8B-Omni \
    --question-file $ROOT/question.json \
    --answer-file $ROOT/answer.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \
    --s2s
python omni_speech/infer/convert_jsonl_to_txt.py $ROOT/answer.json $ROOT/answer.unit
python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
    --in-code-file $ROOT/answer.unit \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
    --results-path $ROOT/answer_wav/ --dur-prediction
