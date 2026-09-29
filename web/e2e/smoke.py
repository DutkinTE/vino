from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = ROOT / "e2e" / "smoke.png"


def main() -> None:
    mock_result = {
        "top1": "Красное вино",
        "top5": [
            {
                "classname": "Красное вино",
                "similarity": 0.92,
                "product": {
                    "id": 12,
                    "wine_name": "Красное вино",
                    "category": "wine",
                    "color": "red",
                    "region": "Италия",
                    "grape_variety": "Sangiovese",
                    "description": "Сухое красное вино",
                    "winery": "Винодельня",
                    "slug": "krasnoe-vino",
                    "photo_name": None,
                    "created_at": None,
                    "updated_at": None,
                },
            }
        ],
        "mode": "label",
        "bottle_found": True,
        "label_found": True,
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        page.add_init_script(
            """
            Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { configurable: true, get: () => 640 });
            Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { configurable: true, get: () => 960 });
            HTMLMediaElement.prototype.play = () => Promise.resolve();
            navigator.mediaDevices = navigator.mediaDevices || {};
            navigator.mediaDevices.getUserMedia = async () => new MediaStream();
            """
        )
        page.route("**/predict", lambda route: route.fulfill(status=200, content_type="application/json", body=__import__("json").dumps(mock_result, ensure_ascii=False)))
        page.goto("http://127.0.0.1:5173", wait_until="networkidle")
        page.get_by_role("button", name="Включить камеру").click()
        page.get_by_label("Предпросмотр камеры").wait_for(state="visible")
        assert page.locator("text=Красное вино").count() == 0
        page.get_by_role("button", name="Сделать снимок").click()
        page.get_by_text("Красное вино", exact=True).wait_for(state="visible")
        page.get_by_text("Найдено · Сходство 0.92", exact=True).wait_for(state="visible")
        print("result_card_background=" + page.locator(".result-card").evaluate("element => getComputedStyle(element).backgroundColor"))
        print("result_card_color=" + page.locator(".result-card").evaluate("element => getComputedStyle(element).color"))
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        print("browser_smoke=passed")
        browser.close()


if __name__ == "__main__":
    main()
