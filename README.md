# Whisper Small Vi V1.1: Whisper Small for Vietnamese Fine-Tuned by Nam Phung 🚀

## 📝 Introduction

This is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) model on Vietnamese speech data. The model aims to improve transcription accuracy and robustness for Vietnamese automatic speech recognition (ASR) tasks, especially in real-world scenarios.

## 📊 Fine-tuning Results

- **Word Error Rate (WER)**: 9.9060

> Evaluation was performed on a held-out test set with diverse regional accents and speaking styles.

## 📝 Model Description

The Whisper small model is a transformer-small sequence-to-sequence model designed for automatic speech recognition and translation tasks. It has been trained on over 680,000 hours of labeled audio data in multiple languages. The fine-tuned version of this model focuses on the Vietnamese language, aiming to improve transcription accuracy and handling of local dialects.

This model works with the WhisperProcessor to pre-process audio inputs into log-Mel spectrograms and decode them into text.

## 📁 Dataset

- Total Duration: More 250 hours of high-quality Vietnamese speech data
- Sources: Public Vietnamese datasets
- Format: 16kHz WAV files with corresponding text transcripts  
- Preprocessing: Audio was normalized and segmented. Transcripts were cleaned and tokenized.  

## 🚀 How to Use

To use the fine-tuned model, you can follow these steps:

1. Install the required dependencies:
   ```python
   # Install required libraries
   !pip install transformers torch librosa soundfile --quiet

   # Import necessary libraries
   import torch
   import librosa
   import soundfile as sf
   from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

   print("Environment setup completed!")
   ```

2. Use the model for inference:
   ```python
   import torch
   import librosa
   from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")

   # Load processor and model
   model_id = "namphungdn134/whisper-small-vi"
   print(f"Loading model from: {model_id}")
   processor = AutoProcessor.from_pretrained(model_id)
   model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)

   # config language and task
   forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
   model.config.forced_decoder_ids = forced_decoder_ids
   print(f"Forced decoder IDs for Vietnamese: {forced_decoder_ids}")

   # Preprocess
   audio_path = "example.wav"  
   print(f"Loading audio from: {audio_path}")
   audio, sr = librosa.load(audio_path, sr=16000)  
   input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
   print(f"Input features shape: {input_features.shape}")

   # Generate
   print("Generating transcription...")
   with torch.no_grad():
      predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

   transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
   print("📝 Transcription:", transcription)

   # Debug: Print token to check
   print("Predicted IDs:", predicted_ids[0].tolist())
   ```

## ⚠️ Limitations

- This model is specifically fine-tuned for the Vietnamese language. It might not perform well on other languages.
- Struggles with overlapping speech or noisy background.
- Performance may drop with strong dialectal variations not well represented in training data.

## 📄 License

This model is licensed under the [MIT License](LICENSE).

## 📚 Citation

If you use this model in your research or application, please cite the original Whisper model and this fine-tuning work as follows:

```
@article{Whisper2021,
  title={Whisper: A Multilingual Speech Recognition Model},
  author={OpenAI},
  year={2021},
  journal={arXiv:2202.12064},
  url={https://arxiv.org/abs/2202.12064}
}
```

```
@misc{title={Whisper small Vi V1.1 - Nam Phung},
  author={Nam Phùng},
  organization={DUT},
  year={2025},
  url={https://huggingface.co/namphungdn134/whisper-small-vi}
}
```

---

## 📬 Contact 

For questions, collaborations, or suggestions, feel free to reach out via [namphungdn134@gmail.com].
