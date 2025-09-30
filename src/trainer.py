import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple

from .model import create_model, create_loss_function
from .data_loader import create_data_loaders
from .utils import save_checkpoint, load_checkpoint, create_directories


class VocalRemoverTrainer:
    """Trainer class for vocal removal model"""
    
    def __init__(self, config: dict, data_paths: List[str], resume_from: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            data_paths: List of paths to training data
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.data_paths = data_paths
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.output_dir = Path(config['paths']['output_dir'])
        self.models_dir = Path(config['paths']['models_dir'])
        self.logs_dir = Path(config['paths']['logs_dir'])
        create_directories([self.output_dir, self.models_dir, self.logs_dir])
        
        # Initialize model
        self.model = create_model(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize loss function
        self.criterion = create_loss_function(config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['training']['patience'] // 2,
            verbose=True
        )
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=self.logs_dir)
        
        # Create data loaders
        print("Creating data loaders...")
        self.train_loader, self.val_loader = create_data_loaders(data_paths, config)
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        num_epochs = self.config['training']['num_epochs']
        save_every = self.config['training']['save_every']
        patience = self.config['training']['patience']
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save model
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = self.models_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(checkpoint_path, epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        print("Training completed!")
        self.writer.close()
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            mixture = batch['mixture'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(mixture)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{avg_loss:.6f}'})
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Batch_Loss/Train', loss.item(), 
                                     len(self.train_loader) * self.start_epoch + batch_idx)
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                mixture = batch['mixture'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                output = self.model(mixture)
                
                # Compute loss
                loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                progress_bar.set_postfix({'Loss': f'{avg_loss:.6f}'})
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.models_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {self.start_epoch}")


class ModelEvaluator:
    """Evaluate trained vocal removal model"""
    
    def __init__(self, model_path: str, config: dict):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def evaluate_dataset(self, data_paths: List[str]) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        from .data_loader import create_data_loaders
        
        # Create data loader
        _, val_loader = create_data_loaders(data_paths, self.config)
        
        # Initialize metrics
        total_loss = 0.0
        num_samples = 0
        
        criterion = create_loss_function(self.config)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                mixture = batch['mixture'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                output = self.model(mixture)
                
                # Compute loss
                loss = criterion(output, target)
                
                total_loss += loss.item() * mixture.size(0)
                num_samples += mixture.size(0)
        
        avg_loss = total_loss / num_samples
        
        return {
            'average_loss': avg_loss,
            'num_samples': num_samples
        }
    
    def separate_audio(self, audio_path: str, output_path: str):
        """Separate vocals from a single audio file"""
        from .data_loader import create_inference_dataset
        from .utils import spectrogram_to_audio
        
        # Create dataset for this audio file
        dataset = create_inference_dataset(audio_path, self.config)
        
        # Process all chunks
        separated_chunks = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                mixture = sample['mixture'].unsqueeze(0).to(self.device)
                
                # Separate vocals
                output = self.model(mixture)
                
                # Convert back to audio
                audio_chunk = spectrogram_to_audio(
                    output.squeeze(0).cpu(),
                    self.config['audio']
                )
                
                separated_chunks.append(audio_chunk)
        
        # Concatenate chunks
        if separated_chunks:
            separated_audio = torch.cat(separated_chunks, dim=1)
            
            # Save output
            import torchaudio
            torchaudio.save(
                output_path,
                separated_audio,
                self.config['audio']['sample_rate']
            )
            
            print(f"Separated audio saved to {output_path}")


def train_model(config_path: str, data_paths: List[str], resume_from: Optional[str] = None):
    """Train vocal removal model"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = VocalRemoverTrainer(config, data_paths, resume_from)
    
    # Start training
    trainer.train()


def evaluate_model(model_path: str, config_path: str, data_paths: List[str]):
    """Evaluate trained model"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, config)
    
    # Evaluate
    results = evaluator.evaluate_dataset(data_paths)
    
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    return results