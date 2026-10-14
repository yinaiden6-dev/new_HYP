"""Start the existing box editor without re-exporting historical dataset lists."""
import argparse
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dataset/tools'))
from photo_metadata import PhotoStore
from review_photos import ReviewServer, Handler


DEEP_LINK = '''<script>
(() => {
  const params = new URLSearchParams(location.search);
  const packageId = params.get('package');
  const photoId = params.get('photo');
  if (!packageId || !photoId) return;
  let attempts = 0;
  const timer = setInterval(() => {
    if (++attempts > 600) { clearInterval(timer); return; }
    const app = document.getElementById('app');
    if (!app || app.hidden || typeof window.openPackage !== 'function') return;
    window.openPackage(packageId, true);
    const button = [...document.querySelectorAll('.subject-box-button')]
      .find(element => element.dataset.photo === photoId);
    if (!button) return;
    clearInterval(timer);
    button.click();
  }, 100);
})();
</script>'''


class FocusedHandler(Handler):
    def do_GET(self):
        if urlsplit(self.path).path == '/':
            if not self.local_host():
                return self.send(dict(error='Use a localhost port forward'), status=403)
            page = (ROOT / 'dataset/tools/photo_review.html').read_text()
            return self.send(page.replace('</body>', DEEP_LINK + '\n</body>').encode(), 'text/html; charset=utf-8')
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ReviewServer(('127.0.0.1', args.port), PhotoStore(ROOT / 'dataset'))
    server.RequestHandlerClass = FocusedHandler
    print(f'Manual subject-box review http://127.0.0.1:{args.port}/?scope=all&package=package_114&photo=photo_c81d2a7abb5af160', flush=True)
    print('Dataset list export skipped; boxes change only after a user save.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
