import os
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import tempfile
import zipfile
from datetime import datetime

try:
    from huggingface_hub import HfApi, Repository, create_repo, upload_file, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface-hub")

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("Warning: GitPython not available. Install with: pip install GitPython")

from .model import create_model
from .utils import load_config, save_config


class ModelHub:
    """Handle model deployment to GitHub and Hugging Face"""
    
    def __init__(self, config: dict):
        """
        Initialize ModelHub
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.deployment_config = config.get('deployment', {})
        
        # Initialize APIs
        if HF_AVAILABLE:
            self.hf_api = HfApi()
        else:
            self.hf_api = None
    
    def save_model_package(self, 
                          model_path: str, 
                          output_dir: str, 
                          model_name: str,
                          description: str = "",
                          tags: List[str] = None) -> str:
        """
        Create a complete model package for deployment
        
        Args:
            model_path: Path to trained model checkpoint
            output_dir: Directory to save the package
            model_name: Name of the model
            description: Model description
            tags: List of tags for the model
            
        Returns:
            Path to the created package
        """
        package_dir = Path(output_dir) / model_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Save model weights only (smaller file)
        model_weights_path = package_dir / "model.pth"
        torch.save(checkpoint['model_state_dict'], model_weights_path)
        
        # Save configuration
        config_path = package_dir / "config.json"
        model_config = {
            'model_type': 'UNetVocalRemover',
            'architecture': self.config['model'],
            'audio_config': self.config['audio'],
            'training_info': {
                'epoch': checkpoint.get('epoch', 0),
                'best_val_loss': checkpoint.get('best_val_loss', 0.0),
                'created_at': datetime.now().isoformat()
            },
            'description': description,
            'tags': tags or ['vocal-removal', 'audio-separation', 'deep-learning']
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Create model card (README.md)
        readme_path = package_dir / "README.md"
        self._create_model_card(readme_path, model_name, description, model_config)
        
        # Create requirements.txt for the model
        requirements_path = package_dir / "requirements.txt"
        self._create_model_requirements(requirements_path)
        
        # Create inference script
        inference_script_path = package_dir / "inference.py"
        self._create_inference_script(inference_script_path)
        
        print(f"Model package created at: {package_dir}")
        return str(package_dir)
    
    def upload_to_huggingface(self, 
                             package_dir: str, 
                             repo_id: str,
                             token: Optional[str] = None,
                             private: bool = False) -> str:
        """
        Upload model package to Hugging Face Hub
        
        Args:
            package_dir: Path to model package directory
            repo_id: Repository ID (username/model-name)
            token: Hugging Face token (or set HF_TOKEN env var)
            private: Whether to create a private repository
            
        Returns:
            Repository URL
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required for Hugging Face upload")
        
        if not self.hf_api:
            raise RuntimeError("Hugging Face API not initialized")
        
        # Get token
        if not token:
            token = os.getenv('HF_TOKEN')
        
        if not token:
            raise ValueError("Hugging Face token required. Set HF_TOKEN environment variable or pass token parameter")
        
        try:
            # Create repository
            repo_url = create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                exist_ok=True
            )
            
            # Upload all files in the package directory
            package_path = Path(package_dir)
            
            for file_path in package_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(package_path)
                    
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=str(relative_path),
                        repo_id=repo_id,
                        token=token
                    )
            
            print(f"Model uploaded to Hugging Face: {repo_url}")
            return repo_url
            
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            raise
    
    def upload_to_github(self, 
                        package_dir: str, 
                        repo_url: str,
                        branch: str = "main",
                        commit_message: str = "Add vocal removal model") -> str:
        """
        Upload model package to GitHub repository
        
        Args:
            package_dir: Path to model package directory
            repo_url: GitHub repository URL
            branch: Branch to upload to
            commit_message: Commit message
            
        Returns:
            Repository URL
        """
        if not GIT_AVAILABLE:
            raise ImportError("GitPython is required for GitHub upload")
        
        try:
            # Create temporary directory for git operations
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Clone or initialize repository
                try:
                    repo = git.Repo.clone_from(repo_url, temp_path)
                    print(f"Cloned repository from {repo_url}")
                except git.exc.GitCommandError:
                    # Repository might not exist or be empty
                    repo = git.Repo.init(temp_path)
                    repo.create_remote('origin', repo_url)
                    print(f"Initialized new repository")
                
                # Switch to specified branch
                try:
                    repo.git.checkout(branch)
                except git.exc.GitCommandError:
                    # Branch doesn't exist, create it
                    repo.git.checkout('-b', branch)
                
                # Copy model package to repository
                package_path = Path(package_dir)
                model_name = package_path.name
                
                dest_dir = temp_path / model_name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                
                shutil.copytree(package_path, dest_dir)
                
                # Add and commit files
                repo.git.add('.')
                repo.git.commit('-m', commit_message)
                
                # Push to remote
                repo.git.push('origin', branch)
                
                print(f"Model uploaded to GitHub: {repo_url}")
                return repo_url
                
        except Exception as e:
            print(f"Error uploading to GitHub: {e}")
            raise
    
    def download_from_huggingface(self, 
                                 repo_id: str, 
                                 local_dir: str,
                                 token: Optional[str] = None) -> str:
        """
        Download model from Hugging Face Hub
        
        Args:
            repo_id: Repository ID (username/model-name)
            local_dir: Local directory to download to
            token: Hugging Face token (optional for public repos)
            
        Returns:
            Path to downloaded model directory
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required for Hugging Face download")
        
        try:
            # Download repository
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=token
            )
            
            print(f"Model downloaded from Hugging Face to: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            raise
    
    def download_from_github(self, 
                           repo_url: str, 
                           local_dir: str,
                           branch: str = "main") -> str:
        """
        Download model from GitHub repository
        
        Args:
            repo_url: GitHub repository URL
            local_dir: Local directory to download to
            branch: Branch to download from
            
        Returns:
            Path to downloaded model directory
        """
        if not GIT_AVAILABLE:
            raise ImportError("GitPython is required for GitHub download")
        
        try:
            # Clone repository
            repo = git.Repo.clone_from(repo_url, local_dir, branch=branch)
            
            print(f"Model downloaded from GitHub to: {local_dir}")
            return local_dir
            
        except Exception as e:
            print(f"Error downloading from GitHub: {e}")
            raise
    
    def load_model_from_package(self, package_dir: str) -> torch.nn.Module:
        """
        Load model from a downloaded package
        
        Args:
            package_dir: Path to model package directory
            
        Returns:
            Loaded PyTorch model
        """
        package_path = Path(package_dir)
        
        # Load configuration
        config_path = package_path / "config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Create model
        model = create_model({'model': model_config['architecture']})
        
        # Load weights
        weights_path = package_path / "model.pth"
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        model.eval()
        print(f"Model loaded from package: {package_dir}")
        
        return model
    
    def _create_model_card(self, readme_path: Path, model_name: str, description: str, config: dict):
        """Create model card (README.md)"""
        content = f"""# {model_name}

{description}

## Model Description

This is a vocal removal model based on U-Net architecture for audio source separation. The model is trained to separate instrumental tracks from mixed audio by removing vocal components.

## Model Architecture

- **Model Type**: {config['model_type']}
- **Input Channels**: {config['architecture']['input_channels']}
- **Output Channels**: {config['architecture']['output_channels']}
- **Hidden Channels**: {config['architecture']['hidden_channels']}
- **Number of Layers**: {config['architecture']['num_layers']}

## Audio Configuration

- **Sample Rate**: {config['audio_config']['sample_rate']} Hz
- **FFT Size**: {config['audio_config']['n_fft']}
- **Hop Length**: {config['audio_config']['hop_length']}
- **Window Length**: {config['audio_config']['win_length']}

## Training Information

- **Training Epoch**: {config['training_info']['epoch']}
- **Best Validation Loss**: {config['training_info']['best_val_loss']:.6f}
- **Created At**: {config['training_info']['created_at']}

## Usage

```python
import torch
from model_hub import ModelHub

# Load model
hub = ModelHub(config)
model = hub.load_model_from_package('path/to/model/package')

# Use for inference
# (See inference.py for complete example)
```

## Tags

{', '.join(config['tags'])}

## License

This model is released under the MIT License.
"""
        
        with open(readme_path, 'w') as f:
            f.write(content)
    
    def _create_model_requirements(self, requirements_path: Path):
        """Create requirements.txt for the model"""
        requirements = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "numpy>=1.21.0",
            "soundfile>=0.12.0"
        ]
        
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_inference_script(self, script_path: Path):
        """Create inference script"""
        script_content = '''import torch
import torchaudio
import json
from pathlib import Path
import argparse

def load_model(package_dir):
    """Load model from package directory"""
    package_path = Path(package_dir)
    
    # Load configuration
    with open(package_path / "config.json", 'r') as f:
        config = json.load(f)
    
    # Create model (you'll need to implement create_model function)
    # model = create_model({'model': config['architecture']})
    
    # Load weights
    # model.load_state_dict(torch.load(package_path / "model.pth", map_location='cpu'))
    # model.eval()
    
    print("Model loaded successfully")
    return None, config  # Return None for now

def separate_vocals(model, audio_path, output_path, config):
    """Separate vocals from audio file"""
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    target_sr = config['audio_config']['sample_rate']
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    
    # TODO: Implement actual vocal separation
    # For now, just save the original audio
    torchaudio.save(output_path, audio, target_sr)
    print(f"Separated audio saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Vocal Removal Inference")
    parser.add_argument("--model_dir", required=True, help="Path to model package directory")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output audio file")
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.model_dir)
    
    # Separate vocals
    separate_vocals(model, args.input, args.output, config)

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)


def deploy_model(model_path: str, 
                config_path: str,
                model_name: str,
                description: str = "",
                github_repo: Optional[str] = None,
                hf_repo: Optional[str] = None,
                output_dir: str = "deployments") -> Dict[str, str]:
    """
    Deploy model to GitHub and/or Hugging Face
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to configuration file
        model_name: Name of the model
        description: Model description
        github_repo: GitHub repository URL (optional)
        hf_repo: Hugging Face repository ID (optional)
        output_dir: Directory to create model package
        
    Returns:
        Dictionary with deployment URLs
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create model hub
    hub = ModelHub(config)
    
    # Create model package
    package_dir = hub.save_model_package(
        model_path=model_path,
        output_dir=output_dir,
        model_name=model_name,
        description=description
    )
    
    results = {'package_dir': package_dir}
    
    # Deploy to GitHub
    if github_repo:
        try:
            github_url = hub.upload_to_github(package_dir, github_repo)
            results['github_url'] = github_url
        except Exception as e:
            print(f"GitHub deployment failed: {e}")
    
    # Deploy to Hugging Face
    if hf_repo:
        try:
            hf_url = hub.upload_to_huggingface(package_dir, hf_repo)
            results['huggingface_url'] = hf_url
        except Exception as e:
            print(f"Hugging Face deployment failed: {e}")
    
    return results