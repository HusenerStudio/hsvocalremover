# Vocal Remover - Deep Learning Audio Separation

A powerful vocal removal system that uses deep learning to separate instrumental tracks from mixed audio. The system supports custom model training, deployment to GitHub/Hugging Face, and high-quality vocal separation.

## Features

- 🎵 **Custom Model Training**: Train your own vocal separation models on custom datasets
- 📁 **Flexible Input**: Accept individual audio files or entire folders for training
- 🧠 **U-Net Architecture**: State-of-the-art deep learning model for audio separation
- 🚀 **Model Deployment**: Upload trained models to GitHub repositories or Hugging Face Hub
- 📥 **Model Download**: Download and use pre-trained models from various sources
- ⚡ **Real-time Processing**: Efficient inference pipeline with chunked processing
- 🎛️ **Configurable**: Extensive configuration options for training and inference

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hsvocalremover
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch, torchaudio, librosa; print('Installation successful!')"
```

## Quick Start

### Training a Model

1. **Prepare your dataset**: Place audio files in a directory structure like:
```
data/
├── song1.wav
├── song2.mp3
├── subfolder/
│   ├── song3.flac
│   └── song4.wav
```

2. **Configure training parameters** (optional - modify `config.yaml`):
```yaml
training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
```

3. **Start training**:
```python
from src.trainer import train_model

train_model(
    config_path="config.yaml",
    data_paths=["data/"],  # List of directories or files
    resume_from=None  # Optional: path to checkpoint to resume from
)
```

### Using a Trained Model for Vocal Removal

```python
from src.inference import VocalRemover

# Initialize vocal remover
vocal_remover = VocalRemover(
    model_path="models/best_model.pth",
    config_path="config.yaml"  # Optional if config is in checkpoint
)

# Remove vocals from a single file
vocal_remover.separate_vocals(
    input_path="input_song.wav",
    output_path="output_instrumental.wav"
)

# Batch process multiple files
vocal_remover.batch_process(
    input_dir="input_songs/",
    output_dir="output_instrumentals/"
)
```

### Command Line Usage

**Training:**
```bash
python -m src.trainer --config config.yaml --data data/ --output models/
```

**Inference:**
```bash
# Single file
python -m src.inference --model models/best_model.pth --input song.wav --output instrumental.wav

# Batch processing
python -m src.inference --model models/best_model.pth --input songs/ --output instrumentals/ --batch
```

## Dataset Specifications

### Supported Audio Formats
- WAV (recommended for training)
- MP3
- FLAC
- M4A
- AAC
- OGG

### Dataset Requirements

**For Training:**
- **Minimum duration**: 1 second per file (recommended: 30+ seconds)
- **Sample rate**: Any (will be resampled to 44.1kHz by default)
- **Channels**: Mono or stereo (stereo recommended)
- **Quality**: Higher quality audio produces better results
- **Quantity**: More diverse data improves model generalization

**Dataset Structure:**
```
training_data/
├── mixed_audio/          # Audio files with vocals
│   ├── song1.wav
│   ├── song2.wav
│   └── ...
└── instrumental/         # Optional: ground truth instrumentals
    ├── song1_inst.wav
    ├── song2_inst.wav
    └── ...
```

**Note**: The current implementation uses a simplified approach for creating training targets. For best results, provide paired vocal/instrumental tracks or use a dataset like MUSDB18.

### Recommended Datasets

1. **MUSDB18**: Professional music separation dataset
2. **DSD100**: Diverse music separation dataset  
3. **Custom datasets**: Your own music collection

## Training Parameters and Configuration

### Key Configuration Options

```yaml
# Model Architecture
model:
  input_channels: 2      # Stereo input
  output_channels: 2     # Stereo output
  hidden_channels: 64    # Model capacity
  num_layers: 6          # Network depth

# Audio Processing
audio:
  sample_rate: 44100     # Target sample rate
  n_fft: 2048           # FFT window size
  hop_length: 512       # Hop length for STFT
  chunk_duration: 10.0   # Training chunk duration (seconds)

# Training Parameters
training:
  batch_size: 8          # Batch size (adjust based on GPU memory)
  learning_rate: 0.001   # Learning rate
  num_epochs: 100        # Maximum epochs
  patience: 10           # Early stopping patience
  validation_split: 0.2  # Validation data ratio

# Loss Function
loss:
  type: "spectral_convergence"  # Loss type
  alpha: 1.0            # Spectral convergence weight
  beta: 0.1             # Magnitude loss weight
```

### Training Tips

1. **GPU Memory**: Reduce `batch_size` if you encounter out-of-memory errors
2. **Training Time**: Expect 1-10 hours depending on dataset size and hardware
3. **Validation**: Monitor validation loss to prevent overfitting
4. **Data Augmentation**: Enable augmentation for better generalization
5. **Checkpoints**: Models are saved every 10 epochs and when validation improves

## Model Export/Import Procedures

### Exporting Models

#### 1. Create Model Package
```python
from src.model_hub import ModelHub

hub = ModelHub(config)
package_dir = hub.save_model_package(
    model_path="models/best_model.pth",
    output_dir="deployments",
    model_name="my_vocal_remover",
    description="Custom vocal removal model trained on pop music"
)
```

#### 2. Deploy to Hugging Face
```python
from src.model_hub import deploy_model

results = deploy_model(
    model_path="models/best_model.pth",
    config_path="config.yaml",
    model_name="my_vocal_remover",
    hf_repo="username/my-vocal-remover"  # Your HF repo
)
```

#### 3. Deploy to GitHub
```python
results = deploy_model(
    model_path="models/best_model.pth",
    config_path="config.yaml", 
    model_name="my_vocal_remover",
    github_repo="https://github.com/username/my-vocal-remover.git"
)
```

### Importing Models

#### From Hugging Face
```python
from src.model_hub import ModelHub

hub = ModelHub(config)
model_dir = hub.download_from_huggingface(
    repo_id="username/my-vocal-remover",
    local_dir="downloaded_models/my_model"
)

# Load the model
model = hub.load_model_from_package(model_dir)
```

#### From GitHub
```python
model_dir = hub.download_from_github(
    repo_url="https://github.com/username/my-vocal-remover.git",
    local_dir="downloaded_models/my_model"
)
```

## Usage Instructions for Vocal Removal

### Basic Usage

```python
from src.inference import VocalRemover

# Initialize
remover = VocalRemover("path/to/model.pth")

# Remove vocals
remover.separate_vocals("input.wav", "output.wav")
```

### Advanced Options

```python
# Custom processing parameters
remover.separate_vocals(
    input_path="input.wav",
    output_path="output.wav",
    chunk_duration=15.0,      # Longer chunks for better quality
    overlap_ratio=0.5,        # More overlap for smoother transitions
    normalize_output=True,    # Normalize output volume
    apply_fade_in_out=True    # Apply fade to prevent clicks
)
```

### Batch Processing

```python
# Process entire directory
results = remover.batch_process(
    input_dir="songs/",
    output_dir="instrumentals/",
    file_extensions=['.wav', '.mp3', '.flac']
)

# Check results
for result in results:
    if 'error' not in result:
        print(f"Processed: {result['input_path']}")
        print(f"Real-time factor: {result['real_time_factor']:.2f}x")
```

### Performance Optimization

1. **GPU Usage**: Automatically uses GPU if available
2. **Chunk Size**: Larger chunks = better quality, more memory
3. **Overlap**: More overlap = smoother output, slower processing
4. **Batch Size**: For multiple files, process in parallel when possible

## API Reference

### Core Classes

- **`VocalRemover`**: Main inference class
- **`VocalRemoverTrainer`**: Training pipeline
- **`ModelHub`**: Model deployment and download
- **`AudioDataset`**: Dataset handling

### Key Functions

- **`train_model()`**: Train a new model
- **`separate_vocals()`**: Remove vocals from audio
- **`deploy_model()`**: Deploy model to repositories
- **`load_model_from_package()`**: Load downloaded model

## Examples

### Example 1: Training on Custom Dataset

```python
import yaml
from src.trainer import train_model

# Load and modify config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['training']['batch_size'] = 4  # Reduce for limited GPU memory
config['training']['num_epochs'] = 50

# Save modified config
with open('custom_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Train model
train_model(
    config_path="custom_config.yaml",
    data_paths=["my_music_collection/"],
    resume_from=None
)
```

### Example 2: Deploying Trained Model

```python
from src.model_hub import deploy_model
import os

# Set Hugging Face token
os.environ['HF_TOKEN'] = 'your_hf_token_here'

# Deploy to both GitHub and Hugging Face
results = deploy_model(
    model_path="models/best_model.pth",
    config_path="config.yaml",
    model_name="pop_vocal_remover",
    description="Vocal remover trained on pop music dataset",
    github_repo="https://github.com/yourusername/pop-vocal-remover.git",
    hf_repo="yourusername/pop-vocal-remover"
)

print(f"GitHub: {results.get('github_url')}")
print(f"Hugging Face: {results.get('huggingface_url')}")
```

### Example 3: Using Pre-trained Model

```python
from src.model_hub import ModelHub
from src.inference import VocalRemover

# Download model from Hugging Face
hub = ModelHub({})
model_dir = hub.download_from_huggingface(
    repo_id="username/vocal-remover-model",
    local_dir="models/downloaded"
)

# Use for inference
remover = VocalRemover(f"{model_dir}/model.pth")
remover.separate_vocals("my_song.wav", "instrumental.wav")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce `batch_size` in config
   - Use smaller `chunk_duration`
   - Use CPU instead of GPU

2. **Poor Separation Quality**:
   - Train longer or with more data
   - Adjust loss function parameters
   - Use higher quality training data

3. **Slow Processing**:
   - Use GPU if available
   - Increase `chunk_duration`
   - Reduce `overlap_ratio`

4. **Model Loading Errors**:
   - Check file paths
   - Ensure config matches model architecture
   - Verify PyTorch version compatibility

### Performance Tips

- **GPU**: Use CUDA-capable GPU for 10-50x speedup
- **Memory**: 8GB+ RAM recommended for training
- **Storage**: SSD recommended for faster data loading
- **CPU**: Multi-core CPU helps with data preprocessing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- librosa developers for audio processing utilities
- Hugging Face for model hosting infrastructure
- Music separation research community

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vocal_remover_2024,
  title={Deep Learning Vocal Remover},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hsvocalremover}
}
```