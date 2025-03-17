# NP Whisper Base Vietnamese - Finetuned by Nam Phung

Model in Huggingface: (https://huggingface.co/namphungdn134/np-whisper-base-vi)

This model is a fine-tuned version of [openai/whisper-base](https://huggingface.co/openai/whisper-base) on the vlsp2020_vinai_100h dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4049
- Wer: 20.3964

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 3500
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Wer     |
|:-------------:|:------:|:----:|:---------------:|:-------:|
| 0.7425        | 0.1492 | 250  | 0.7303          | 34.7993 |
| 0.6393        | 0.2983 | 500  | 0.6105          | 30.0462 |
| 0.5636        | 0.4475 | 750  | 0.5404          | 28.1320 |
| 0.5199        | 0.5967 | 1000 | 0.5043          | 25.5525 |
| 0.4806        | 0.7458 | 1250 | 0.4756          | 24.4785 |
| 0.4779        | 0.8950 | 1500 | 0.4581          | 23.8864 |
| 0.414         | 1.0442 | 1750 | 0.4447          | 22.6037 |
| 0.3967        | 1.1933 | 2000 | 0.4336          | 21.2506 |
| 0.3723        | 1.3425 | 2250 | 0.4243          | 21.8426 |
| 0.3886        | 1.4916 | 2500 | 0.4179          | 21.3605 |
| 0.3876        | 1.6408 | 2750 | 0.4128          | 20.8728 |
| 0.3459        | 1.7900 | 3000 | 0.4086          | 20.7572 |
| 0.3546        | 1.9391 | 3250 | 0.4058          | 20.5909 |
| 0.3208        | 2.0883 | 3500 | 0.4049          | 20.3964 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.4.0
- Tokenizers 0.21.1
