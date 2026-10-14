"""Serve the package photo editor on a loopback port."""
import argparse
import base64
import io
import json
import mimetypes
import os
import secrets
from functools import lru_cache, cached_property
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from PIL import Image, ImageOps
from photo_metadata import ROOT, Conflict, PhotoStore, sha
from subject_boxes import SubjectBoxes
from confusion_review import ConfusionReview


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        super().__init__(address, Handler)
        self.store = store
        self.boxes = SubjectBoxes(store)
        # Keep open editors usable across a server restart.
        with store.locked():
            token_path = store.review / '.session_token'
            if not token_path.exists():
                fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    stream.write(secrets.token_urlsafe(32))
            self.token = token_path.read_text().strip()
        self.thumbs = {}
        self.pairs = []
        previous = store.root / 'review/duplicates/index.html'
        report = store.root / 'review/duplicates/report.json'
        if report.exists():
            data = json.loads(report.read_text())
            if data['dataset_sha256'] == sha((store.root / 'lists/photos.csv').read_bytes()):
                self.pairs = [dict(left=data['photos'][p['left']]['photo_id'],
                                  right=data['photos'][p['right']]['photo_id'],
                                  priority=p['priority'], score=p['score'],
                                  cross_split=p['cross_split']) for p in data['pairs']]
                if previous.exists():
                    text = previous.read_text()
                    marker = '<script type="application/json" id="review-data">'
                    if marker in text:
                        embedded = json.loads(text.split(marker, 1)[1].split('</script>', 1)[0])
                        self.thumbs = {p['photo_id']: base64.b64decode(p['thumb'].split(',', 1)[1])
                                       for p in embedded['photos']}

    @cached_property
    def confusion(self):
        return ConfusionReview(self.store)

    @lru_cache(maxsize=256)
    def reference_thumbnail(self, reference_id):
        row = self.confusion.all_refs[reference_id]
        with Image.open(self.store.root / row['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((700, 700))
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=88)
            return stream.getvalue()

    @lru_cache(maxsize=256)
    def thumbnail(self, photo_id):
        if photo_id in self.thumbs:
            return self.thumbs[photo_id]
        with Image.open(self.store.root / self.store.photos[photo_id]['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((640, 640))
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=85)
            return stream.getvalue()

    def package_view(self, package_id):
        package, revision = self.store.package(package_id)
        names = list(dict.fromkeys(p['name'] for d in package['documents'] for p in d.get('products', [])))
        return dict(package_id=package_id, name=' / '.join(names) or package['name'],
                    design_name=package['name'], category=package['category'],
                    split=self.store.roles[package_id], revision=revision,
                    photos=[dict(p, photo_id=Path(p['image']).stem) for p in package['photos']])

    def review_scopes(self):
        scopes = [dict(id='all', name='全部照片', photo_ids=list(self.store.photos))]
        release = self.store.root / 'release.json'
        supplement = json.loads(release.read_text()).get('supplement') if release.exists() else None
        if supplement:
            receipt = (self.store.root / supplement).resolve()
            if not receipt.is_relative_to(self.store.root / 'review'):
                raise ValueError('Supplement receipt must be inside dataset/review')
            ids = json.loads(receipt.read_text())['photo_ids']
            if not isinstance(ids, list) or not all(isinstance(pid, str) for pid in ids):
                raise ValueError('Invalid supplement photo IDs')
            if len(ids) != len(set(ids)) or not set(ids).issubset(self.store.photos):
                raise ValueError('Supplement photos differ from the current inventory')
            if ids:
                scopes.append(dict(id='supplement', name='新增补充照片', photo_ids=ids))
        return scopes


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Photo IDs and request bodies are intentionally absent from the access log.
        pass

    def send(self, data, content_type='application/json; charset=utf-8', status=200, cache=False):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'private, max-age=3600' if cache else 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; connect-src 'self'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Navigation can cancel an image request while the server prepares it.
            pass

    def local_host(self):
        return urlsplit('http://' + self.headers.get('Host', '')).hostname in ('localhost', '127.0.0.1', '::1')

    def do_GET(self):
        if not self.local_host():
            return self.send(dict(error='Use a localhost port forward'), status=403)
        path = urlsplit(self.path).path.rstrip('/') or '/'
        try:
            if path == '/confusion':
                return self.send(Path(__file__).with_name('confusion_review.html').read_bytes(), 'text/html; charset=utf-8')
            if path in ('/confusion_review.js', '/confusion_review.css'):
                return self.send(Path(__file__).with_name(path[1:]).read_bytes(),
                                 'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
            if path == '/api/confusion/state':
                return self.send(dict(token=self.server.token, **self.server.confusion.summary()))
            if path == '/api/confusion/export':
                return self.send(self.server.confusion.export())
            if path.startswith('/api/confusion/package/'):
                return self.send(self.server.confusion.package(path.rsplit('/', 1)[1]))
            if path == '/api/confusion/browse':
                query = parse_qs(urlsplit(self.path).query)
                one = lambda key, default: query.get(key, [default])[0]
                return self.send(self.server.confusion.browse(query=one('q', ''), scope=one('scope', 'all'),
                    offset=int(one('offset', '0')), limit=int(one('limit', '36')), package_id=one('package', None),
                    status=one('status', 'all')))
            if path.startswith('/reference-thumb/'):
                return self.send(self.server.reference_thumbnail(path.rsplit('/', 1)[1]), 'image/jpeg', cache=True)
            if path.startswith('/reference-image/'):
                row = self.server.confusion.all_refs[path.rsplit('/', 1)[1]]
                image = self.server.store.root / row['path']
                return self.send(image.read_bytes(), mimetypes.guess_type(image.name)[0] or 'application/octet-stream', cache=True)
            if path in ('/', '/review/photos', '/review/photos/index.html', '/review/duplicates', '/review/duplicates/index.html'):
                return self.send((Path(__file__).with_name('photo_review.html')).read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/state':
                with self.server.store.locked():
                    packages = [self.server.package_view(p) for p in self.server.store.packages]
                return self.send(dict(token=self.server.token, packages=packages, pairs=self.server.pairs,
                                      review_scopes=self.server.review_scopes()))
            if path in ('/subject_boxes.js', '/subject_boxes.css'):
                return self.send(Path(__file__).with_name(path[1:]).read_bytes(),
                                 'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
            if path == '/api/boxes':
                return self.send(self.server.boxes.summary())
            if path.startswith('/api/box/'):
                return self.send(self.server.boxes.get(path.rsplit('/', 1)[1]))
            if path.startswith('/annotation-image/'):
                identifier = path.rsplit('/', 1)[1]
                self.server.boxes.image_info(identifier)
                with Image.open(self.server.store.root / self.server.store.photos[identifier]['path']) as source:
                    stream = io.BytesIO()
                    ImageOps.exif_transpose(source).convert('RGB').save(stream, 'JPEG', quality=95)
                return self.send(stream.getvalue(), 'image/jpeg', cache=True)
            if path.startswith('/api/package/'):
                return self.send(self.server.package_view(path.rsplit('/', 1)[1]))
            kind, identifier = path.strip('/').split('/')
            if kind == 'thumb':
                return self.send(self.server.thumbnail(identifier), 'image/jpeg', cache=True)
            if kind == 'image':
                image = self.server.store.root / self.server.store.photos[identifier]['path']
            elif kind == 'design':
                image = self.server.store.root / self.server.store.packages[identifier]['design_path']
            else:
                raise KeyError('Unknown route')
            return self.send(image.read_bytes(), mimetypes.guess_type(image.name)[0] or 'application/octet-stream', cache=True)
        except ValueError as error:
            return self.send(dict(error=str(error)), status=400)
        except (KeyError, FileNotFoundError):
            return self.send(dict(error='Not found'), status=404)

    def do_POST(self):
        origin = self.headers.get('Origin')
        same_origin = not origin or urlsplit(origin).netloc == self.headers.get('Host')
        if not self.local_host() or not same_origin or self.headers.get('X-Review-Token') != self.server.token:
            return self.send(dict(error='Invalid editor session; reload the page'), status=403)
        path = urlsplit(self.path).path
        if not path.startswith(('/api/package/', '/api/box/', '/api/confusion/')):
            return self.send(dict(error='Not found'), status=404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 100_000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(size))
            if path == '/api/confusion/pair':
                return self.send(self.server.confusion.update_pair(body))
            if path.startswith('/api/confusion/package/'):
                return self.send(self.server.confusion.update_package(path.rsplit('/', 1)[1], body))
            if path.startswith('/api/confusion/design/'):
                return self.send(self.server.confusion.update_design(path.rsplit('/', 1)[1], body))
            if path.startswith('/api/confusion/'):
                return self.send(dict(error='Not found'), status=404)
            if path.startswith('/api/box/'):
                return self.send(self.server.boxes.update(path.rsplit('/', 1)[1], body))
            if not isinstance(body, dict) or set(body) != {'revision', 'changes'} or not isinstance(body['revision'], str):
                raise ValueError('Expected revision and changes')
            package_id = path.rsplit('/', 1)[1]
            self.server.store.update(package_id, body['revision'], body['changes'])
            return self.send(self.server.package_view(package_id))
        except Conflict as error:
            return self.send(dict(error=str(error)), status=409)
        except (ValueError, KeyError, TypeError) as error:
            return self.send(dict(error=str(error)), status=400)
        except OSError:
            return self.send(dict(error='Could not finish saving. Reload and check the saved state.'), status=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    store = PhotoStore(args.root)
    store.export_active()
    server = ReviewServer(('127.0.0.1', args.port), store)
    scope = 'supplement' if any(s['id'] == 'supplement' for s in server.review_scopes()) else 'all'
    print(f'Photo review: http://127.0.0.1:{args.port}/?scope={scope}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
