import torch
from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.training import create_trainer
from alphaearth.data import create_aef_dataloader, create_aef_dataloader_from_npz


def main():
    # Use synthetic data to sanity-check training end-to-end
    dl = create_aef_dataloader(num_samples=2, batch_size=1, num_workers=0, num_frames=4, patch_size=64)

    model = AlphaEarthFoundations(model_size="small")
    trainer = create_trainer(model, dl, None, output_dir="./outputs_small")

    trainer.max_steps = 10
    trainer.warmup_steps = 0
    trainer.train(max_steps=trainer.max_steps, log_every=1)
    print("Training run finished.")


if __name__ == "__main__":
    main()
