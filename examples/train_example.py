#!/usr/bin/env python3
"""
Example training script for vocal removal model

This script demonstrates how to:
1. Set up training configuration
2. Prepare training data
3. Train a vocal removal model
4. Monitor training progress
5. Deploy the trained model

Usage:
    python examples/train_example.py --data_dir path/to/audio/files --output_dir models/
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.trainer import VocalRemoverTrainer
from src.model_hub import deploy_model
from src.utils import create_directories, validate_audio_file


def setup_training_config(args):
    """Create training configuration"""
    
    config = {
        # Model Architecture
        'model': {
            'name': 'UNetVocalRemover',
            'input_channels': 2,
            'output_channels': 2,
            'hidden_channels': 64,
            'num_layers': 6,
            'kernel_size': 3,
            'stride': 2
        },
        
        # Audio Processing
        'audio': {
            'sample_rate': 44100,
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'chunk_duration': args.chunk_duration,
            'overlap': 0.25
        },
        
        # Training Parameters
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'patience': args.patience,
            'validation_split': 0.2,
            'save_every': 10
        },
        
        # Loss Function
        'loss': {
            'type': args.loss_type,
            'alpha': 1.0,
            'beta': 0.1
        },
        
        # Data Augmentation
        'augmentation': {
            'enabled': args.augmentation,
            'pitch_shift_range': [-2, 2],
            'time_stretch_range': [0.9, 1.1],
            'noise_factor': 0.01
        },
        
        # Paths
        'paths': {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'models_dir': os.path.join(args.output_dir, 'models'),
            'logs_dir': os.path.join(args.output_dir, 'logs')
        }
    }
    
    return config


def validate_data_directory(data_dir):
    """Validate training data directory"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(data_path.rglob(f'*{ext}')))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    # Validate some files
    valid_files = 0
    for audio_file in audio_files[:10]:  # Check first 10 files
        if validate_audio_file(str(audio_file)):
            valid_files += 1
    
    if valid_files == 0:
        raise ValueError("No valid audio files found")
    
    print(f"Found {len(audio_files)} audio files in {data_dir}")
    print(f"Validated {valid_files}/10 sample files")
    
    return audio_files


def main():
    parser = argparse.ArgumentParser(description="Train vocal removal model")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True, 
                       help="Directory containing training audio files")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory for models and logs")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--chunk_duration", type=float, default=10.0,
                       help="Audio chunk duration in seconds")
    
    # Training options
    parser.add_argument("--loss_type", default="spectral_convergence",
                       choices=["spectral_convergence", "l1", "mse", "combined"],
                       help="Loss function type")
    parser.add_argument("--augmentation", action="store_true",
                       help="Enable data augmentation")
    parser.add_argument("--resume_from", 
                       help="Path to checkpoint to resume training from")
    
    # Deployment options
    parser.add_argument("--deploy_hf", 
                       help="Deploy to Hugging Face (provide repo_id)")
    parser.add_argument("--deploy_github",
                       help="Deploy to GitHub (provide repo_url)")
    parser.add_argument("--model_name", default="vocal_remover",
                       help="Name for the deployed model")
    
    args = parser.parse_args()
    
    print("=== Vocal Remover Training Example ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    
    try:
        # Validate data directory
        print("\n--- Validating Data ---")
        audio_files = validate_data_directory(args.data_dir)
        
        # Create output directories
        print("\n--- Setting Up Output Directories ---")
        create_directories([
            Path(args.output_dir),
            Path(args.output_dir) / 'models',
            Path(args.output_dir) / 'logs'
        ])
        
        # Setup configuration
        print("\n--- Creating Configuration ---")
        config = setup_training_config(args)
        
        # Save configuration
        config_path = Path(args.output_dir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to: {config_path}")
        
        # Create trainer
        print("\n--- Initializing Trainer ---")
        trainer = VocalRemoverTrainer(
            config=config,
            data_paths=[args.data_dir],
            resume_from=args.resume_from
        )
        
        # Start training
        print("\n--- Starting Training ---")
        print("Training will begin now. Monitor progress in TensorBoard:")
        print(f"tensorboard --logdir {config['paths']['logs_dir']}")
        print("\nPress Ctrl+C to stop training early (model will be saved)")
        
        try:
            trainer.train()
            print("\n--- Training Completed Successfully ---")
            
        except KeyboardInterrupt:
            print("\n--- Training Interrupted by User ---")
            print("Saving current model state...")
            
            # Save current state
            current_epoch = trainer.start_epoch
            checkpoint_path = Path(config['paths']['models_dir']) / f"interrupted_epoch_{current_epoch}.pth"
            trainer.save_checkpoint(checkpoint_path, current_epoch, is_best=False)
            print(f"Model saved to: {checkpoint_path}")
        
        # Find best model
        models_dir = Path(config['paths']['models_dir'])
        best_model_path = models_dir / "best_model.pth"
        
        if best_model_path.exists():
            print(f"\n--- Best Model Available ---")
            print(f"Best model: {best_model_path}")
            
            # Deploy model if requested
            if args.deploy_hf or args.deploy_github:
                print("\n--- Deploying Model ---")
                
                try:
                    results = deploy_model(
                        model_path=str(best_model_path),
                        config_path=str(config_path),
                        model_name=args.model_name,
                        description=f"Vocal removal model trained on custom dataset",
                        github_repo=args.deploy_github,
                        hf_repo=args.deploy_hf
                    )
                    
                    print("Deployment results:")
                    for key, value in results.items():
                        print(f"  {key}: {value}")
                        
                except Exception as e:
                    print(f"Deployment failed: {e}")
                    print("You can deploy manually later using the model_hub module")
            
            # Print usage instructions
            print(f"\n--- Usage Instructions ---")
            print("To use your trained model for vocal removal:")
            print(f"")
            print(f"from src.inference import VocalRemover")
            print(f"")
            print(f"remover = VocalRemover('{best_model_path}')")
            print(f"remover.separate_vocals('input.wav', 'output.wav')")
            print(f"")
            print("Or use the command line:")
            print(f"python -m src.inference --model {best_model_path} --input song.wav --output instrumental.wav")
            
        else:
            print("\n--- No Best Model Found ---")
            print("Training may have been interrupted early or failed.")
            print("Check the models directory for available checkpoints:")
            print(f"ls {models_dir}")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTraining failed. Please check your data and configuration.")
        return 1
    
    print("\n=== Training Example Completed ===")
    return 0


if __name__ == "__main__":
    exit(main())