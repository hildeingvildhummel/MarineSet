import os
import torch
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForPreTraining,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import numpy as np

# =====================
# Configuration
# =====================
MODEL_NAME = "facebook/wav2vec2-base"
DATA_DIR = "/projects/0/vusr0637/test/"
OUTPUT_DIR = "models/wav2vec2-scratch32kHz"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-5

# =====================
# Load model + feature extractor
# =====================
model = Wav2Vec2ForPreTraining.from_pretrained(MODEL_NAME)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# Optional: increase masking probability
model.config.mask_time_prob = 0.075
model.config.mask_time_length = 20
model.config.apply_spec_augment = True
model.config.num_negatives = 100

model.train()

# =====================
# Prepare dataset
# =====================

audio_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".flac")]
data_dict = {"path": audio_files}
dataset = Dataset.from_dict(data_dict)

print(f"Dataset ready with {len(dataset)} audio files")

from torch.utils.data import IterableDataset
import librosa
import soundfile as sf

class AudioDataset(IterableDataset):
    def __init__(self, files, feature_extractor):
        self.files = files
        self.feature_extractor = feature_extractor

    def __iter__(self):
        for path in self.files:
            try:
                audio, sr = sf.read(path, dtype="float32")

                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                if len(audio) == 0:
                    print(f"Skipping empty file: {path}")
                    continue

                if sr != 16000:
                    audio = librosa.resample(
                        audio,
                        orig_sr=sr,
                        target_sr=16000,
                    )

                audio = np.asarray(audio, dtype=np.float32)

                # -------------------------
                # Split into 5-second windows
                # -------------------------
                TARGET_SR = 16000
                SEGMENT_LENGTH = 5  # seconds
                NUM_SAMPLES = TARGET_SR * SEGMENT_LENGTH

                num_windows = len(audio) // NUM_SAMPLES

                for i in range(num_windows):

                    start = i * NUM_SAMPLES
                    end = start + NUM_SAMPLES

                    audio_section = audio[start:end]

                    # -------------------------
                    # Feature extractor
                    # -------------------------

                    inputs = self.feature_extractor(
                        audio_section,
                        sampling_rate=TARGET_SR,
                    )

                    input_values = np.asarray(
                        inputs["input_values"][0],
                        dtype=np.float32
                    )

                    if len(input_values) == 0:
                        continue

                    yield {
                        "input_values": input_values
                    }

                # equivalent of dataset.filter()
                if len(input_values) == 0:
                    print(f"Skipping empty features: {path}")
                    continue

                yield {
                    "input_values": input_values
                }

            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

dataset = AudioDataset(
    audio_files,
    feature_extractor
)

# =====================
# Data collator
# =====================
# Transformers 5.2.0 Wav2Vec2ForPreTraining automatically handles masking & negatives internally
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://huggingface.co/papers/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    def __init__(self, model, feature_extractor, padding, pad_to_multiple_of=None, mask_time_prob=0.075, mask_time_length=10):
        self.model = model
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length

    def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

data_collator = DataCollatorForWav2Vec2Pretraining(
    model=model,
    feature_extractor=feature_extractor,
    padding=True,
)

# =====================
# Training arguments
# =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LEARNING_RATE,

    max_steps=100000,

    fp16=True,
    logging_steps=50,
    save_total_limit=1,
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=5000,

    report_to="wandb",
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

# =====================
# Trainer
# =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

batch = next(iter(trainer.get_train_dataloader()))
outputs = model(**batch)
print(outputs.keys())

# =====================
# Train
# =====================
trainer.train()

# =====================
# Save final model
# =====================
trainer.save_model(OUTPUT_DIR)
print("Training complete. Model saved.")