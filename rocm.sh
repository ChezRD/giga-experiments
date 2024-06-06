#!/bin/bash

# start=`date +%s`

# python ssl_inference.py --encoder_config ./data/encoder_config.yaml \
#     --model_weights ./data/ssl_model_weights.ckpt --device cuda --audio_path ./data/example.mp3

# python ssl_inference.py --encoder_config ./data/encoder_config.yaml \
#     --model_weights ./data/ssl_model_weights.ckpt --device cuda --audio_path ../downloads/nts.mp3.mp3

# encoded signal shape: torch.Size([1, 768, 159])

# GigaAM-CTC
# python ctc_inference.py --model_config ./data/ctc_model_config.yaml \
#     --model_weights ./data/ctc_model_weights.ckpt --device cuda --audio_path ./data/example.mp3
# https://www.youtube.com/watch?v=xVah87LKS04
# https://www.youtube.com/watch?v=a3wEyzvHYSE
# https://www.youtube.com/watch?v=qHXw1RiX7dM

# LD_PRELOAD="/usr/local/lib/libforcegttalloc.so" HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_SDMA=0 python ctc2_inference.py --model_config ./data/ctc_model_config.yaml \
# LD_PRELOAD="" HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_SDMA=1 python ctc2_inference.py --model_config ./data/ctc_model_config.yaml \
#     --model_weights ./data/ctc_model_weights.ckpt --device xpu --youtube_url https://www.youtube.com/watch?v=qHXw1RiX7dM


# start=`date +%s`
# echo "GigaAM-CTC long-form"
# AMD_LOG_LEVEL=0 LD_PRELOAD="" HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_SDMA=0 python ctc_longform_inference.py --model_config ./data/ctc_model_config.yaml \
#     --model_weights ./data/ctc_model_weights.ckpt --device cuda --fp16 True \
#     --audio_path ../downloads/nts.mp3.mp3 --batch_size 4
# # ./data/long_example.wav
# # ./downloads/nts.wav.wav
# # ../downloads/nts.mp3.mp3
# end=`date +%s`
# echo Execution time was `expr $end - $start` seconds.


# start=`date +%s`
# echo "GigaAM-RNNT long-form"
# AMD_LOG_LEVEL=0 LD_PRELOAD="" HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_SDMA=0 python rnnt_longform_inference.py --model_config ./data/rnnt_model_config.yaml \
#     --tokenizer_path ./data/tokenizer_all_sets \
#     --model_weights ./data/rnnt_model_weights.ckpt --device cuda --fp16 True \
#     --audio_path ../downloads/nts.mp3.mp3 --batch_size 4
# # ./data/long_example.wav
# # ./downloads/nts.wav.wav
# # ../downloads/nts.mp3.mp3
# # ../downloads/output.wav
# end=`date +%s`
# echo Execution time was `expr $end - $start` seconds.


start=`date +%s`
echo "GigaAM-RNNT long-form"
python rnnt_longform_sliding.py --model_config ./data/rnnt_model_config.yaml \
    --tokenizer_path ./data/tokenizer_all_sets \
    --model_weights ./data/rnnt_model_weights.ckpt --device cuda --fp16 True \
    --audio_path ../downloads/output.wav --batch_size 6
# ./data/long_example.wav
# ./downloads/nts.wav.wav
# ../downloads/nts.mp3.mp3
# ../downloads/output.wav
end=`date +%s`
echo Execution time was `expr $end - $start` seconds.


# start=`date +%s`
# echo "GigaAM-CTC long-form"
# python ctc_longform_inference.py --model_config ./data/ctc_model_config.yaml \
#     --model_weights ./data/ctc_model_weights.ckpt --device cuda --fp16 True \
#     --audio_path ./data/long_example.wav --hf_token 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 --batch_size 32

# end=`date +%s`
# echo Execution time was `expr $end - $start` seconds.


# start=`date +%s`
# echo "GigaAM-RNNT long-form"
# python rnnt_longform_inference.py --model_config ./data/rnnt_model_config.yaml \
#     --model_weights ./data/rnnt_model_weights.ckpt --tokenizer_path ./data/tokenizer_all_sets \
#     --device cuda --fp16 True --audio_path ./data/long_example.wav --hf_token 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 --batch_size 32

# end=`date +%s`
# echo Execution time was `expr $end - $start` seconds.

# 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

# transcription: а и правда никакой

# GigaAM-Emo
# python emo_inference.py --model_config ./data/emo_model_config.yaml \
#     --model_weights ./data/emo_model_weights.ckpt --device cuda --audio_path ./data/example.mp3

# python emo_inference.py --model_config ./data/emo_model_config.yaml \
#     --model_weights ./data/emo_model_weights.ckpt --device cuda --audio_path ../downloads/nts.mp3.mp3

# end=`date +%s`

# echo Execution time was `expr $end - $start` seconds.