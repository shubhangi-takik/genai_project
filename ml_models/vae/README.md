# Interior-image VAE

This is a separate, unconditional convolutional VAE. It reconstructs uploaded
images and samples new images from its learned standard-normal latent prior.
It does not accept text prompts or use room/style labels to control generation.
The existing Autoencoder model and checkpoint are unchanged.

The scripts use the dataset's existing `train_data.csv`, `val_data.csv`, and
`test_data.csv` files. Images are converted to RGB, bilinearly resized to a
square, and scaled to `[0, 1]`, matching the existing Autoencoder preprocessing.
Training reads only the train and validation splits; the test split is used only
by the evaluation script.

## Dependencies

No additional packages are required. Use the project's Autoencoder virtual
environment, which already provides PyTorch, NumPy, Pillow, and the existing
evaluation helpers.

## Train

From the project root in PowerShell:

```powershell
& ".\ml_models\autoencoder\.venv\Scripts\python.exe" ".\ml_models\vae\train_vae.py" --data-root "C:\Users\shubhangi\OneDrive\Desktop\gen_ai_project\data\Pinterest Interior Design Images and Metadata" --epochs 30 --batch-size 16 --image-size 128 --latent-dim 128 --beta 1.0 --beta-warmup-epochs 10 --patience 6 --seed 42 --device cpu
```

The best validation checkpoint, epoch log, JSON history, and total,
reconstruction, and KL loss graphs are saved in `outputs/`.

## Evaluate and sample

Run after a successful training run:

```powershell
& ".\ml_models\autoencoder\.venv\Scripts\python.exe" ".\ml_models\vae\evaluate_vae.py" --data-root "C:\Users\shubhangi\OneDrive\Desktop\gen_ai_project\data\Pinterest Interior Design Images and Metadata" --checkpoint ".\ml_models\vae\outputs\best_vae.pt" --existing-checkpoint ".\ml_models\autoencoder\outputs\best_autoencoder.pt" --seed 42
```

The script saves test originals and reconstructions, per-image and summary
metrics, a reconstruction comparison grid, reproducible prior samples, and an
evaluation report under `outputs/evaluation_test/`. Reconstruction metrics
describe fidelity to test inputs; they do not measure the realism or diversity
of sampled images.

## Backend and frontend

Start the backend from the project root:

```powershell
& ".\ml_models\autoencoder\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

The Vite proxy defaults to `http://127.0.0.1:8001`. Set `VITE_BACKEND_URL`
before starting Vite to use a different backend address. The frontend can be
started from `frontend` with `npm run dev`.
The VAE page exposes reconstruction and unconditional sampling when the trained
checkpoint is available. API-request images are saved under
`backend/results/vae/` and served through the existing `/results` mount.
The GAN page also uses backend inference with `ml_models/gan/outputs/best_gan.pt`.
This checkpoint maps structural edge maps to 128 x 128 room images; it is not
conditioned on text prompts or trained to add specific furniture.

The Transformer page's `POST /api/transformer/analyze` endpoint extracts
interior-design terms with a deterministic rule-based baseline. It is not a
trained Transformer NLP model and does not generate images. The frontend uses
the Vite proxy by default; see `frontend/.env.example` for the optional
`VITE_API_BASE_URL` and `VITE_BACKEND_URL` settings.
