import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ApiError, type PredictionResponse, predictImage } from './api';
import { captureVideoFrame, openRearCamera } from './camera';
import App from './App';

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api');
  return { ...actual, predictImage: vi.fn() };
});

vi.mock('./camera', () => ({
  captureVideoFrame: vi.fn(),
  openRearCamera: vi.fn(),
  stopCamera: vi.fn(),
}));

const response: PredictionResponse = {
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

const stream = {
  getTracks: () => [{ stop: vi.fn() }],
} as unknown as MediaStream;

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  vi.mocked(openRearCamera).mockResolvedValue(stream);
  vi.mocked(captureVideoFrame).mockResolvedValue(new Blob(['frame'], { type: 'image/jpeg' }));
  vi.stubGlobal('URL', {
    createObjectURL: vi.fn().mockReturnValue('blob:gallery-preview'),
    revokeObjectURL: vi.fn(),
  });
});

describe('scanner UI', () => {
  it('shows a mobile-only message on desktop instead of the scanner', async () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({
      matches: true,
      media: '(min-width: 720px)',
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));

    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Откройте Vino Scan с телефона.' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Включить камеру' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Выбрать фото')).not.toBeInTheDocument();
  });

  it('shows a camera CTA and gallery fallback initially', () => {
    render(<App />);

    expect(screen.getByRole('button', { name: 'Включить камеру' })).toBeInTheDocument();
    expect(screen.getByLabelText('Выбрать фото')).toBeInTheDocument();
  });

  it('opens the camera without predicting until the user takes a photo', async () => {
    vi.mocked(predictImage).mockResolvedValue(response);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: 'Включить камеру' }));

    expect(await screen.findByLabelText('Предпросмотр камеры')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Сделать снимок' })).toBeInTheDocument();
    expect(predictImage).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Сделать снимок' }));

    expect(await screen.findByText('Красное вино')).toBeInTheDocument();
    expect(screen.getByText('Найдено · Сходство 0.92')).toBeInTheDocument();
    expect(screen.getByText('Италия')).toBeInTheDocument();
  });

  it('shows a recoverable API error after a camera request fails', async () => {
    vi.mocked(predictImage).mockRejectedValue(new ApiError('Сервис временно недоступен', 503));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: 'Включить камеру' }));
    await user.click(screen.getByRole('button', { name: 'Сделать снимок' }));

    expect(await screen.findByText('Модель ещё запускается. Попробуйте через несколько секунд.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Повторить сканирование' })).toBeInTheDocument();
  });

  it('shows the selected gallery photo while recognition is pending', async () => {
    let resolvePrediction!: (value: PredictionResponse) => void;
    vi.mocked(predictImage).mockReturnValue(new Promise((resolve) => {
      resolvePrediction = resolve;
    }));
    const user = userEvent.setup();
    render(<App />);
    const file = new File(['image'], 'wine.jpg', { type: 'image/jpeg' });

    await user.upload(screen.getByLabelText('Выбрать фото'), file);

    expect(await screen.findByAltText('Выбранное фото бутылки')).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('Анализирую фото…');

    resolvePrediction(response);
    expect(await screen.findByText('Красное вино')).toBeInTheDocument();
  });

  it('uses the same prediction flow for a gallery image', async () => {
    vi.mocked(predictImage).mockResolvedValue(response);
    const user = userEvent.setup();
    render(<App />);
    const file = new File(['image'], 'wine.jpg', { type: 'image/jpeg' });

    await user.upload(screen.getByLabelText('Выбрать фото'), file);

    expect(predictImage).toHaveBeenCalledWith(expect.any(Blob));
    expect(await screen.findByText('Красное вино')).toBeInTheDocument();
  });
});
