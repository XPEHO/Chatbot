import json
import urllib.error
import urllib.request


class BookStackClient:
    def __init__(self, base_url, token_id, token_secret):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Token {token_id}:{token_secret}"}

    def _get(self, url):
        req = urllib.request.Request(url, headers=self.headers, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def build_url(self, path_or_url):
        return f"{self.base_url}{path_or_url}"
        

    def list_pages(self, url=None):
        path = "/api/pages"
        url = self.build_url(path)
        status, body = self._get(url)
        if status >= 400:
            raise RuntimeError(
                f"HTTP {status} for {url}: {body.decode('utf-8', errors='replace')}"
            )
        return json.loads(body.decode("utf-8"))

    def export_page_pdf(self, page_id):
        path = f"/api/pages/{page_id}/export/pdf"
        url = self.build_url(path)
        status, body = self._get(url)
        if status >= 400:
            raise RuntimeError(
                f"HTTP {status} for {url}: {body.decode('utf-8', errors='replace')}"
            )
        return body
