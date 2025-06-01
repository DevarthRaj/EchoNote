import whisper
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Trainer, TrainingArguments

def prepare_dataset(batch):
    # Prepare audio and transcription pairs
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch

# Initialize model and processor
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load your custom dataset
# Replace "your_dataset" with actual dataset path/name
dataset = load_dataset("your_dataset")

# Prepare dataset
processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./whisper-fine-tuned")