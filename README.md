# giga-model-experiments

Experiments with https://github.com/salute-developers/GigaAM models

Test files - https://drive.google.com/drive/folders/11RMVQYvEPho5OuYgg2qI7FW2MVDgSr3L

requirements: ffmpeg, yt-dlp

```bash
mkdir data
cd data

wget https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/{ssl_model_weights.ckpt,emo_model_weights.ckpt,ctc_model_weights.ckpt,rnnt_model_weights.ckpt,ctc_model_config.yaml,emo_model_config.yaml,encoder_config.yaml,rnnt_model_config.yaml,tokenizer_all_sets.tar,example.wav,long_example.wav}
tar -xf tokenizer_all_sets.tar && rm tokenizer_all_sets.tar

```
