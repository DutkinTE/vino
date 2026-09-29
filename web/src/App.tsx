import { useEffect, useRef, useState, type ChangeEvent, type ReactElement } from 'react';
import { ApiError, predictImage, type PredictionResponse } from './api';
import { captureVideoFrame, openRearCamera, stopCamera } from './camera';

const MAX_FRAME_WIDTH = 1280;
const DESKTOP_MEDIA_QUERY = '(min-width: 720px)';

type ScannerStatus = 'idle' | 'starting' | 'ready' | 'recognizing' | 'error';

function messageForError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 503) {
      return 'Модель ещё запускается. Попробуйте через несколько секунд.';
    }
    if (error.status === 400) {
      return 'Кадр не удалось прочитать. Наведите камеру на этикетку ещё раз.';
    }
    return error.message;
  }
  if (error instanceof DOMException && error.name === 'NotAllowedError') {
    return 'Доступ к камере запрещён. Разрешите камеру в настройках браузера или выберите фото.';
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'Не удалось распознать бутылку. Попробуйте ещё раз.';
}

function statusCopy(status: ScannerStatus, hasPreview: boolean): string {
  if (status === 'starting') return 'Запускаю камеру…';
  if (status === 'recognizing') return hasPreview ? 'Анализирую фото…' : 'Ищу совпадение…';
  if (status === 'error') return 'Попробуйте ещё раз';
  return 'Наведите камеру на этикетку';
}

function displayValue(value: string | null): string | null {
  return value?.trim() || null;
}

function useDesktopViewport(): boolean {
  const [isDesktop, setIsDesktop] = useState(() => (
    typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia(DESKTOP_MEDIA_QUERY).matches
  ));

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const mediaQuery = window.matchMedia(DESKTOP_MEDIA_QUERY);
    const updateViewport = (): void => setIsDesktop(mediaQuery.matches);
    updateViewport();
    mediaQuery.addEventListener('change', updateViewport);
    return () => mediaQuery.removeEventListener('change', updateViewport);
  }, []);

  return isDesktop;
}

function DesktopGate(): ReactElement {
  return (
    <main className="app-shell desktop-gate-shell">
      <div className="ambient-glow ambient-glow-one" />
      <div className="ambient-glow ambient-glow-two" />
      <header className="brand-bar">
        <div>
          <span className="brand-kicker">Vino / scan</span>
          <span className="brand-tagline">Ваш карманный сомелье</span>
        </div>
        <span className="live-pill"><span className="live-dot" /> мобильный режим</span>
      </header>

      <section className="desktop-gate-card" aria-label="Откройте сервис с телефона">
        <div className="phone-orbit" aria-hidden="true">
          <span className="phone-spark phone-spark-one">✦</span>
          <span className="phone-spark phone-spark-two">·</span>
          <span className="phone-frame">
            <span className="phone-speaker" />
            <span className="phone-screen"><span>✦</span></span>
            <span className="phone-home" />
          </span>
        </div>
        <span className="eyebrow">Vino Scan создан для телефона</span>
        <h1>Откройте Vino Scan<br /><em>с телефона.</em></h1>
        <p>Снимите этикетку камерой смартфона — сервис найдёт вино и покажет его описание.</p>
        <div className="desktop-gate-steps">
          <div><span>01</span><strong>Откройте эту ссылку на телефоне</strong></div>
          <div><span>02</span><strong>Наведите камеру на этикетку</strong></div>
          <div><span>03</span><strong>Получите результат</strong></div>
        </div>
      </section>
    </main>
  );
}

function ProductFacts({ result }: { result: PredictionResponse }): ReactElement | null {
  const product = result.top5[0]?.product;
  if (!product) return null;
  const facts = [
    ['Цвет', displayValue(product.color)],
    ['Регион', displayValue(product.region)],
    ['Сорт', displayValue(product.grape_variety)],
    ['Винодельня', displayValue(product.winery)],
  ].filter((fact): fact is [string, string] => Boolean(fact[1]));

  return (
    <>
      <dl className="result-details-grid">
        <div className="product-detail product-detail-name">
          <dt>Название</dt>
          <dd>{product.wine_name}</dd>
        </div>
        {facts.map(([label, value]) => (
          <div className="product-detail" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

function ResultCard({ result }: { result: PredictionResponse }): ReactElement {
  const best = result.top5[0];
  return (
    <section
      className="result-card"
      aria-live="polite"
      aria-label={`Результат распознавания: ${best?.product.wine_name ?? result.top1}`}
    >
      <ProductFacts result={result} />
    </section>
  );
}

export default function App(): ReactElement {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const sessionRef = useRef(0);
  const autoCameraStartedRef = useRef(false);
  const isDesktop = useDesktopViewport();
  const [status, setStatus] = useState<ScannerStatus>('idle');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const stopSession = (): void => {
    stopCamera(streamRef.current);
    streamRef.current = null;
  };

  const clearPreview = (): void => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
  };

  const showPreview = (file: Blob): void => {
    const nextPreviewUrl = URL.createObjectURL(file);
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = nextPreviewUrl;
    setPreviewUrl(nextPreviewUrl);
  };

  useEffect(() => () => {
    stopSession();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);
  useEffect(() => {
    if (isDesktop) stopSession();
  }, [isDesktop]);

  const startCamera = async (): Promise<void> => {
    const video = videoRef.current;
    if (!video) return;
    const session = sessionRef.current + 1;
    sessionRef.current = session;
    stopSession();
    clearPreview();
    setError(null);
    setResult(null);
    setStatus('starting');

    try {
      const stream = await openRearCamera(video);
      if (sessionRef.current !== session) {
        stopCamera(stream);
        return;
      }
      streamRef.current = stream;
      setStatus('ready');
    } catch (nextError) {
      if (sessionRef.current !== session) return;
      stopSession();
      setError(messageForError(nextError));
      setStatus('error');
    }
  };

  useEffect(() => {
    if (isDesktop) {
      autoCameraStartedRef.current = false;
      return;
    }
    if (autoCameraStartedRef.current) return;
    autoCameraStartedRef.current = true;
    void startCamera();
  }, [isDesktop]);

  const capturePhoto = async (): Promise<void> => {
    const currentVideo = videoRef.current;
    const session = sessionRef.current;
    if (!currentVideo || !streamRef.current) {
      setError('Предпросмотр камеры недоступен');
      setStatus('error');
      return;
    }

    setError(null);
    setStatus('recognizing');
    try {
      const frame = await captureVideoFrame(currentVideo, MAX_FRAME_WIDTH);
      if (sessionRef.current !== session) return;
      showPreview(frame);
      stopSession();
      const nextResult = await predictImage(frame);
      if (sessionRef.current !== session) return;
      setResult(nextResult);
      setStatus('ready');
    } catch (nextError) {
      if (sessionRef.current !== session) return;
      setError(messageForError(nextError));
      setStatus('error');
    }
  };

  const resetScan = (): void => {
    setError(null);
    setResult(null);
    void startCamera();
  };

  const handleGallery = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    const session = sessionRef.current + 1;
    sessionRef.current = session;
    stopSession();
    setResult(null);
    showPreview(file);
    setStatus('recognizing');
    setError(null);
    try {
      const nextResult = await predictImage(file);
      if (sessionRef.current !== session) return;
      setResult(nextResult);
      setStatus('ready');
    } catch (nextError) {
      if (sessionRef.current !== session) return;
      setError(messageForError(nextError));
      setStatus('error');
    }
  };

  if (isDesktop) return <DesktopGate />;

  const hasCamera = Boolean(streamRef.current);
  return (
    <main className="app-shell scanner-shell">
      <section
        className={`camera-stage ${result ? 'has-result' : ''}`}
        aria-label="Область сканирования"
        aria-busy={status === 'recognizing'}
      >
        <video
          ref={videoRef}
          className={`camera-video ${previewUrl ? 'is-covered' : ''}`}
          aria-label="Предпросмотр камеры"
          aria-hidden={Boolean(previewUrl)}
          playsInline
          muted
          autoPlay
        />
        {previewUrl && (
          <img className="captured-image" src={previewUrl} alt="Выбранное фото бутылки" />
        )}
        {!previewUrl && !hasCamera && <div className="camera-placeholder" aria-hidden="true"><span>Фото бутылки<br />появится здесь</span></div>}
        <div className="viewfinder-veil" aria-hidden="true">
          <span className="viewfinder-blur-panel viewfinder-blur-top" />
          <span className="viewfinder-blur-panel viewfinder-blur-bottom" />
          <span className="viewfinder-blur-panel viewfinder-blur-left" />
          <span className="viewfinder-blur-panel viewfinder-blur-right" />
          <span className="viewfinder-blur-corner viewfinder-blur-corner-tl" />
          <span className="viewfinder-blur-corner viewfinder-blur-corner-tr" />
          <span className="viewfinder-blur-corner viewfinder-blur-corner-bl" />
          <span className="viewfinder-blur-corner viewfinder-blur-corner-br" />
        </div>
        <div className={`scan-frame ${result ? 'is-result' : ''}`} aria-hidden="true">
          <span className="frame-corner frame-corner-tl" />
          <span className="frame-corner frame-corner-tr" />
          <span className="frame-corner frame-corner-bl" />
          <span className="frame-corner frame-corner-br" />
          <span className="frame-line" />
        </div>
        <header className="scanner-header">
          <div
            className="scan-instruction"
            aria-live="polite"
            role={status === 'recognizing' ? 'status' : undefined}
          >
            {status === 'recognizing' ? (
              <span className="loading-spinner" aria-hidden="true" />
            ) : null}
            <span>{statusCopy(status, Boolean(previewUrl))}</span>
          </div>
        </header>
        {result && <ResultCard result={result} />}

        <section className={`control-area ${result ? 'has-result' : ''}`} aria-live="polite">
          {!result && status === 'idle' && (
            <div className="scan-hint">
              <span className="scan-hint-kicker">VINO / SCAN</span>
              <p>Найдите вино по фотографии этикетки</p>
            </div>
          )}

          {!result && status === 'error' && error && (
            <div className="error-card" role="alert">
              <span className="error-icon">!</span>
              <div><strong>Не получилось</strong><p>{error}</p></div>
            </div>
          )}

          {hasCamera && !result && status === 'ready' && (
            <button className="primary-button" type="button" onClick={() => void capturePhoto()}>
              <span className="primary-label">Сканировать заново</span>
              <span className="button-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></svg>
              </span>
            </button>
          )}

          {status === 'starting' && (
            <button className="primary-button" type="button" disabled>
              <span className="primary-label">Подключаем камеру…</span>
              <span className="button-icon" aria-hidden="true"><span className="loading-spinner" /></span>
            </button>
          )}

          {!hasCamera && status !== 'starting' && status !== 'recognizing' && !result && (
            <button className="primary-button" type="button" onClick={() => void startCamera()}>
              <span className="primary-label">{status === 'error' ? 'Повторить сканирование' : 'Включить камеру'}</span>
              <span className="button-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></svg>
              </span>
            </button>
          )}

          {result && (
            <button className="primary-button rescan-button" type="button" onClick={resetScan}>
              <span className="primary-label">Сканировать заново</span>
              <span className="button-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></svg>
              </span>
            </button>
          )}

          <label className="gallery-button">
            <span>Выбрать фото из галереи</span>
            <input aria-label="Выбрать фото" type="file" accept="image/*" onChange={(event) => void handleGallery(event)} />
          </label>
        </section>
      </section>
    </main>
  );
}
