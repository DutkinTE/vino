import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildApiUrl, predictImage } from './api';

const predictionResponse = {
  top1: 'Красное вино',
  top5: [
    {
      classname: 'Красное вино',
      similarity: 0.92,
      product: {
        id: 12,
        wine_name: 'Красное вино',
        category: 'wine',
        color: 'red',
        region: 'Италия',
        grape_variety: 'Sangiovese',
        description: 'Сухое красное вино',
        winery: 'Винодельня',
        slug: 'krasnoe-vino',
        photo_name: null,
        created_at: null,
        updated_at: null,
      },
    },
  ],
  mode: 'label',
  bottle_found: true,
  label_found: true,
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe('predictImage', () => {
  it('builds an absolute URL when the API is hosted separately', () => {
    expect(buildApiUrl('/predict', 'https://api.виносвое.рф/')).toBe('https://api.виносвое.рф/predict');
  });

  it('posts a JPEG blob as multipart data and returns the typed response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(predictionResponse), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const blob = new Blob(['image'], { type: 'image/jpeg' });

    await expect(predictImage(blob)).resolves.toEqual(predictionResponse);

    expect(fetchMock).toHaveBeenCalledWith('/predict', expect.objectContaining({ method: 'POST' }));
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(request.body).toBeInstanceOf(FormData);
    expect((request.body as FormData).get('file')).toBeInstanceOf(File);
  });

  it('turns an HTTP error into an ApiError with a safe message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Не удалось прочитать изображение' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    await expect(predictImage(new Blob(['image'], { type: 'image/jpeg' }))).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      message: 'Не удалось прочитать изображение',
    });
  });
});
