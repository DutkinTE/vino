# Vino Scan web client

Mobile-first React/Vite interface for the bottle recognition API in the parent project.

## Run locally

From the project root, start FastAPI in one terminal:

```bash
source ../.venv-macos/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/predict` and `/health` to `http://127.0.0.1:8080`; the backend intentionally does not enable CORS.

## Phone camera

The camera uses `navigator.mediaDevices.getUserMedia` and prefers the rear camera. It keeps a preview open, captures one frame when the user presses «Сделать снимок», and sends that frame to `/predict`. The gallery picker uses the same API path.

Mobile browsers require a secure context for camera access: use `https://` or `localhost`. A plain LAN URL such as `http://192.168.1.10:5173` may load the page but can block the camera. For a physical-phone smoke test, expose the Vite app through a trusted HTTPS development certificate or an HTTPS reverse proxy. If camera permission is unavailable, use the gallery fallback.

## Production build

```bash
npm run build
```

The output is `web/dist`. A reverse proxy should serve the frontend from one HTTPS origin and forward `/predict` and `/health` to FastAPI. This keeps browser requests same-origin and avoids CORS.

## Checks

```bash
npm test
npm run build
```
