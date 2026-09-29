import unittest

from fastapi.testclient import TestClient

from app.main import app


class CorsTests(unittest.TestCase):
    def test_allows_frontend_preflight_for_api_requests(self) -> None:
        client = TestClient(app)
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://xn--b1aajkzgbw.xn--p1ai",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://xn--b1aajkzgbw.xn--p1ai")


if __name__ == "__main__":
    unittest.main()
