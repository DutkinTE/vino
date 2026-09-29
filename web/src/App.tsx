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

function statusCopy(status: ScannerStatus, hasResult: boolean, hasPreview: boolean): string {
  if (status === 'starting') return 'Запускаю камеру…';
  if (status === 'recognizing') return hasPreview ? 'Анализирую фото…' : 'Ищу совпадение…';
  if (status === 'error') return 'Нужен ещё один кадр';
  if (hasResult) return 'Снимок распознан';
  if (status === 'ready') return 'Проверьте этикетку в рамке';
  return 'Найдём ваше вино по этикетке';
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
      {facts.length > 0 && (
        <dl className="facts-grid">
          {facts.map(([label, value]) => (
            <div className="fact" key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {displayValue(product.description) && <p className="product-description">{product.description}</p>}
    </>
  );
}

function ResultCard({ result, onReset }: { result: PredictionResponse; onReset: () => void }): ReactElement {
  const best = result.top5[0];
  const alternatives = result.top5.slice(1);
  return (
    <section className="result-card" aria-live="polite" aria-label="Результат распознавания">
      <div className="result-heading">
        <span className="eyebrow">{`Найдено · Сходство ${best?.similarity.toFixed(2) ?? '—'}`}</span>
        <span className="result-mode">{result.label_found ? 'Этикетка' : 'Бутылка'}</span>
      </div>
      <h1>{best?.product.wine_name ?? result.top1}</h1>
      <ProductFacts result={result} />
      {alternatives.length > 0 && (
        <details className="alternatives">
          <summary>Ещё совпадения</summary>
          <ul>
            {alternatives.map((match) => (
              <li key={`${match.classname}-${match.similarity}`}>
                <span>{match.product.wine_name}</span>
                <span>{match.similarity.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <button className="secondary-button" type="button" onClick={onReset}>
        Сканировать снова
      </button>
    </section>
  );
}

export default function App(): ReactElement {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const sessionRef = useRef(0);
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
    <main className="app-shell">
      <div className="ambient-glow ambient-glow-one" />
      <div className="ambient-glow ambient-glow-two" />
      <header className="brand-bar">
        <div>
          <span className="brand-kicker">Vino / scan</span>
          <span className="brand-tagline">Ваш карманный сомелье</span>
        </div>
        <span className={`live-pill ${hasCamera ? 'is-live' : ''}`}>
          <span className="live-dot" /> {hasCamera ? 'камера' : 'готово'}
        </span>
      </header>

      <section className="camera-stage" aria-label="Область сканирования" aria-busy={status === 'recognizing'}>
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
        {!previewUrl && <div className="camera-shade" aria-hidden="true" />}
        {!previewUrl && <div className="scan-frame" aria-hidden="true">
          <span className="frame-corner frame-corner-tl" />
          <span className="frame-corner frame-corner-tr" />
          <span className="frame-corner frame-corner-bl" />
          <span className="frame-corner frame-corner-br" />
          <span className="frame-line" />
        </div>}
        {previewUrl && status === 'recognizing' && (
          <div className="photo-loading" role="status">
            <span className="loading-spinner" aria-hidden="true" />
            <span>Анализирую фото…</span>
          </div>
        )}
        <div className={`scan-caption ${status === 'recognizing' ? 'is-loading' : ''}`}>
          <span className="caption-mark">✦</span>
          <span>{statusCopy(status, Boolean(result), Boolean(previewUrl))}</span>
        </div>
      </section>

      <section className="control-area" aria-live="polite">
        {!result && status === 'idle' && (
          <div className="intro-copy">
            <span className="eyebrow">Сканирование этикетки</span>
            <h1>Найдём бутылку<br /><em>по одному кадру.</em></h1>
            <p>Наведите камеру на этикетку — база вин сама подберёт самое близкое совпадение.</p>
          </div>
        )}

        {!result && status === 'error' && error && (
          <div className="error-card" role="alert">
            <span className="error-icon">!</span>
            <div><strong>Не получилось</strong><p>{error}</p></div>
          </div>
        )}

        {result && <ResultCard result={result} onReset={resetScan} />}

        {hasCamera && !result && status === 'ready' && (
          <button className="primary-button" type="button" onClick={() => void capturePhoto()}>
            <span className="button-icon" aria-hidden="true">◉</span>
            Сделать снимок
          </button>
        )}

        {(!hasCamera || status === 'error') && (
          <button className="primary-button" type="button" onClick={() => void startCamera()}>
            <span className="button-icon" aria-hidden="true">◎</span>
            {status === 'error' ? 'Повторить сканирование' : 'Включить камеру'}
          </button>
        )}

        <label className="gallery-button">
          <span>Или выбрать фото из галереи</span>
          <input aria-label="Выбрать фото" type="file" accept="image/*" onChange={(event) => void handleGallery(event)} />
        </label>
      </section>
    </main>
  );
}
