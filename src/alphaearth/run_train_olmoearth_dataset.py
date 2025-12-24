import argparse
from pathlib import Path
import torch
from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.training import create_trainer
from alphaearth.data_olmoearth import create_olmoearth_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train AlphaEarth on OlmoEarth pretrain dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/olmoearth_pretrain_dataset/10_landsat_monthly",
        help="Directory containing tar files",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV metadata file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Spatial patch size (H, W)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides --epochs if set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for learning rate",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Log every N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_olmoearth",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--landsat_bands",
        type=int,
        default=7,
        help="Number of Landsat bands to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading OlmoEarth dataset from {data_dir}")
    print(f"Using {args.landsat_bands} Landsat bands")
    
    dataloader = create_olmoearth_dataloader(
        data_dir=str(data_dir),
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        normalize=True,
        shuffle=True,
        num_bands=args.landsat_bands,
    )
    
    dataset_size = len(dataloader.dataset)
    steps_per_epoch = dataset_size // args.batch_size
    
    if args.max_steps is None:
        max_steps = args.epochs * steps_per_epoch
    else:
        max_steps = args.max_steps
    
    print(f"Dataset: {dataset_size} samples, {steps_per_epoch} steps/epoch")
    print(f"Training for {max_steps} steps ({max_steps / steps_per_epoch:.2f} epochs)")
    
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources={"landsat": args.landsat_bands},
        decode_sources={"landsat": args.landsat_bands},
    )
    
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Model size: small")
    print(f"Input sources: {model.input_sources}")
    print(f"Decode sources: {model.decode_sources}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    param_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size (float32): {param_size_mb:.2f} MB")
    
    print("\n" + "-"*80)
    print("MODEL ARCHITECTURE")
    print("-"*80)
    print(model)
    print("="*80 + "\n")
    
    trainer = create_trainer(
        model=model,
        dataloader=dataloader,
        text_adapter=None,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    trainer.max_steps = max_steps
    trainer.warmup_steps = args.warmup_steps
    
    print(f"Starting training for {args.max_steps} steps...")
    print(f"Output directory: {args.output_dir}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(max_steps=args.max_steps, log_every=args.log_every)
    
    print("Training run finished.")


if __name__ == "__main__":
    main()

