# Bottle recognition API

FastAPI service that follows the inference path from `unified_siglip_bottle_comparison.ipynb`: optionally detects a bottle and label, creates a fine-tuned SigLIP image embedding, and returns the five closest product classes. Similarity is a cosine similarity score, not a probability.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the image-processing pipeline, component boundaries, retrieval backends, and response flow.

## Put model and reference files here

- `models/siglip2_finetuned.pt`: fine-tuned SigLIP `state_dict` checkpoint used by the notebook. It must match `google/siglip2-base-patch16-384`.
- `models/bottle/bottle_detector_best.pt`: optional YOLO `detect` weights for bottle cropping.
- `models/label_pose_best.pt`: optional YOLO `pose` weights for perspective-correcting the label.
- `data/reference/embedding_comparison_reference.csv`: must include `wine_name` and the four 768-dimensional columns `embedding_label_base`, `embedding_full_base`, `embedding_label_finetuned`, and `embedding_full_finetuned`. An optional `product_id` disambiguates catalog entries that share a display name; `filename` is optional.
- `data/reference/product_info.csv`: product catalog with `Название вина`, `Категория`, `Цвет`, `Регион`, `Сорт винограда`, `Описание`, `Винодельня`, `Slug`, and `Название фото`.
- `data/test_images/`: optional local images for testing `/predict/path`.

The two YOLO files are optional. If they are absent, the API embeds the full image and searches `embedding_full_finetuned`. With a bottle detector but no label detection, it searches the bottle crop. When a label is detected, it searches `embedding_label_finetuned`.

Place the CSVs and model weights at the paths above before importing or making predictions. The SigLIP base model is downloaded from Hugging Face on first use unless it is already cached.

## Run locally

Use Python 3.10 or newer. On macOS/Linux, create `.venv-macos`; the existing `.venv` in this copy is a Windows virtual environment. Create `.env` from `.env.example`, then install the packages:

```bash
python3 -m venv .venv-macos
source .venv-macos/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU/CUDA environment using the official PyTorch selector. Start the API from this directory:

```bash
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8080/docs` for interactive API testing. `GET /health` reports whether the model has been loaded.

## Mobile web client

The separate React/Vite client is in `web/`. Start the API and frontend in two terminals:

```bash
source .venv-macos/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
cd web
npm install
npm run dev
```

Open `http://localhost:5173` on the development computer. Vite proxies `/predict` and `/health` to FastAPI, so CORS is not required. The interface keeps a camera preview and sends one JPEG frame only after you press «Сделать снимок». It also supports selecting a photo from the gallery.

Camera access on a physical phone requires `https://` or `localhost`; a plain HTTP LAN address can show the page but block the camera. Use a trusted HTTPS development certificate or reverse proxy for phone testing. See [`web/README.md`](web/README.md) for the frontend details.

## Endpoints

Backend upload, as multipart form data:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/predict" -F "file=@data/test_images/bottle.jpg"
```

Local path test: set `LOCAL_TEST_ENABLED=true` in `.env`, put the image under `data/test_images`, and send:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict/path" `
  -H "Content-Type: application/json" `
  -d '{"path":"data/test_images/bottle.jpg"}'
```

The local-path endpoint only serves files inside `TEST_IMAGES_DIR`; keep it disabled in a public deployment. Both endpoints return JSON in this shape:

```json
{
  "top1": "normalized product name",
  "top5": [
    {
      "classname": "normalized product name",
      "similarity": 0.91,
      "product": {
        "id": 12,
        "wine_name": "normalized product name",
        "category": "wine",
        "color": "red",
        "region": "...",
        "grape_variety": "...",
        "description": "...",
        "winery": "...",
        "slug": "normalized-product-name",
        "photo_name": "bottle.webp",
        "created_at": "2026-09-28T12:00:00",
        "updated_at": "2026-09-28T12:00:00"
      }
    }
  ],
  "mode": "label",
  "bottle_found": true,
  "label_found": true
}
```

`top1` is the predicted value on its own; it is also the first item in `top5`.

## Retrieval backends

With `VECTOR_BACKEND=csv`, the API reads both reference embeddings and product details from the two CSV files. PostgreSQL is not required; this mode is suitable for a small CPU-only server.

With `VECTOR_BACKEND=pgvector`, the API reads product details and vectors from PostgreSQL. Configure `PG_DSN`, apply `db/init.sql`, and import the catalog before its embeddings:

```powershell
python scripts/import_products_csv_to_postgres.py
python scripts/import_csv_to_pgvector.py
```

The product importer copies all named catalog fields to `products`. The embedding importer prefers `product_id` when supplied; otherwise it matches `wine_name` only when that name identifies one catalog row. It writes four records per CSV row to `product_embeddings`: `label` as `crop`, `full` as `full`, with separate `base` and `finetuned` model versions. Missing or ambiguous product names and unknown product IDs are listed in `data/reference/unmatched_product_embeddings.csv`. Repeated source rows for one product/type/version update the existing row because of the table's unique key.

## Tests

The vector retrieval tests do not require model weights or a GPU:

```powershell
python -m unittest discover -s tests -v
```
