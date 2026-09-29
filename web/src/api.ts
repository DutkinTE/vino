export interface ProductDetails {
  id: number | null;
  wine_name: string;
  category: string | null;
  color: string | null;
  region: string | null;
  grape_variety: string | null;
  description: string | null;
  winery: string | null;
  slug: string | null;
  photo_name: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface Prediction {
  classname: string;
  similarity: number;
  product: ProductDetails;
}

export interface PredictionResponse {
  top1: string;
  top5: Prediction[];
  mode: 'label' | 'full' | string;
  bottle_found: boolean;
  label_found: boolean;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  vector_backend: string;
  local_test_enabled: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

const configuredApiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? '').trim();

export function buildApiUrl(path: string, baseUrl = configuredApiBaseUrl): string {
  const normalizedBase = baseUrl.replace(/\/+$/, '');
  const normalizedPath = `/${path.replace(/^\/+/, '')}`;
  return `${normalizedBase}${normalizedPath}`;
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch {
    throw new ApiError('Не удалось связаться с сервером', 0);
  }

  if (!response.ok) {
    let message = `Сервер вернул ошибку ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === 'string' && body.detail.trim()) {
        message = body.detail;
      }
    } catch {
      // Keep the generic status message when the server did not return JSON.
    }
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as T;
}

export function predictImage(blob: Blob, signal?: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('file', blob, 'camera-frame.jpg');
  return requestJson<PredictionResponse>(buildApiUrl('/predict'), {
    method: 'POST',
    body: form,
    signal,
  });
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>(buildApiUrl('/health'), { signal });
}
