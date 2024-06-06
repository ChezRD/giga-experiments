#!/bin/bash

# Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
# Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
source /opt/intel/oneapi/setvars.sh --force

# Recommended Environment Variables for optimal performance
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE=1
export SYCL_CACHE_PERSISTENT=1
export IPEX_XPU_ONEDNN_LAYOUT=1
export OverrideDefaultFP64Settings=1
export IGC_EnableDPEmulation=1
export ZES_ENABLE_SYSMAN=1
export NEOReadDebugKeys=1
# export ClDeviceGlobalMemSizeAvailablePercent=85
export UseXPU=true
export SYCL_CACHE_PERSISTENT=1
export IPEX_LOGGING=1

start=`date +%s`
echo "GigaAM-CTC long-form"
python ctc_longform_inference.py --model_config ./data/ctc_model_config.yaml \
    --model_weights ./data/ctc_model_weights.ckpt --device xpu --fp16 True \
    --audio_path ./downloads/nts.wav.wav --batch_size 4
# ./data/long_example.wav
# ./downloads/nts.wav.wav
# ../downloads/nts.mp3.mp3
end=`date +%s`
echo Execution time was `expr $end - $start` seconds.

# start=`date +%s`
# echo "GigaAM-RNNT long-form"
# python rnnt_longform_inference.py --model_config ./data/rnnt_model_config.yaml \
#     --tokenizer_path ./data/tokenizer_all_sets \
#     --model_weights ./data/rnnt_model_weights.ckpt --device xpu --fp16 True \
#     --audio_path ./downloads/nts.wav.wav --batch_size 2
# # ./data/long_example.wav
# # ./downloads/nts.wav.wav
# # ../downloads/nts.mp3.mp3
# # ../downloads/output.wav
# end=`date +%s`
# echo Execution time was `expr $end - $start` seconds.


# start=`date +%s`
# echo "GigaAM-RNNT long-form"
# python rnnt_longform_sliding.py --model_config ./data/rnnt_model_config.yaml \
#     --tokenizer_path ./data/tokenizer_all_sets \
#     --model_weights ./data/rnnt_model_weights.ckpt --device xpu --fp16 True \
#     --audio_path ./downloads/nts.wav.wav --batch_size 24
# # ./data/long_example.wav
# # ./downloads/nts.wav.wav
# # ../downloads/nts.mp3.mp3
# # ../downloads/output.wav
# end=`date +%s`
# echo Execution time was `expr $end - $start` seconds.

# start=`date +%s`
# echo "GigaAM-RNNT long-form"
# python rnnt_longform_inference.py --model_config ./data/rnnt_model_config.yaml \
#     --model_weights ./data/rnnt_model_weights.ckpt --tokenizer_path ./data/tokenizer_all_sets \
#     --device xpu --fp16 True --audio_path ./downloads/nts.wav.wav --hf_token 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 --batch_size 32

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