"""Serve deduplicated retrieval-error review on a localhost port."""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlsplit

from PIL import Image
from error_analysis import Analysis, Conflict, ROOT


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, analysis):
        super().__init__(address, Handler)
        self.analysis = analysis
        with analysis.locked():
            path = analysis.review / '.session_token'
            if not path.exists():
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    stream.write(secrets.token_urlsafe(32))
            self.token = path.read_text().strip()

    @lru_cache(maxsize=128)
    def image(self, kind, identifier, cid='', full=False):
        store = self.analysis
        if kind == 'photo':
            path = store.safe_path('dataset/' + store.photos[identifier]['path'])
        elif kind == 'reference':
            path = store.safe_path('dataset/' + store.refs[identifier]['path'])
        elif kind == 'input':
            path = store.safe_path(store.processed[cid][identifier]['query_path'])
        else:
            raise ValueError('Unknown image kind')
        with Image.open(path) as source:
            orientation = source.getexif().get(274, 1)
            image = source.convert('RGB')
            method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                      4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                      6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                      8: Image.Transpose.ROTATE_90}.get(orientation)
            if method is not None:
                image = image.transpose(method)
            if not full:
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            # Avoid serializing malformed, unrelated source EXIF tags.
            image.save(stream, 'JPEG', quality=94)
            return stream.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, mime='application/json; charset=utf-8', status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; connect-src 'self'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local(self):
        return urlsplit('http://' + self.headers.get('Host', '')).hostname in ('localhost', '127.0.0.1', '::1')

    def do_GET(self):
        if not self.local():
            return self.send(dict(error='Use localhost port forwarding'), status=403)
        request = urlsplit(self.path)
        path = request.path.rstrip('/') or '/'
        try:
            if path == '/':
                return self.send(Path(__file__).with_name('error_review.html').read_bytes(), 'text/html; charset=utf-8')
            if path in ('/error_review.js', '/error_review.css'):
                mime = 'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8'
                return self.send(Path(__file__).with_name(path[1:]).read_bytes(), mime)
            if path == '/api/state':
                return self.send(dict(token=self.server.token, **self.server.analysis.summary()))
            if path == '/api/cases':
                q = parse_qs(request.query)
                one = lambda name, default: q.get(name, [default])[0]
                return self.send(self.server.analysis.browse(config=one('config', 'F03'),
                    status=one('status', 'pending'), pattern=one('pattern', ''), query=one('q', '')))
            if path.startswith('/api/case/'):
                return self.send(self.server.analysis.detail(path.rsplit('/', 1)[1]))
            if path.startswith('/image/'):
                parts = path.strip('/').split('/')
                full = parse_qs(request.query).get('full') == ['1']
                if len(parts) == 3:
                    return self.send(self.server.image(parts[1], parts[2], full=full), 'image/jpeg')
                if len(parts) == 4 and parts[1] == 'input':
                    return self.send(self.server.image('input', parts[3], cid=parts[2], full=full), 'image/jpeg')
                raise ValueError('Invalid image URL')
            return self.send(dict(error='Not found'), status=404)
        except KeyError:
            return self.send(dict(error='Not found'), status=404)
        except ValueError as error:
            return self.send(dict(error=str(error)), status=400)
        except OSError:
            return self.send(dict(error='File unavailable'), status=404)

    def do_POST(self):
        origin = self.headers.get('Origin')
        expected_origin = 'http://' + self.headers.get('Host', '')
        if not self.local() or self.headers.get('X-Review-Token') != self.server.token or (origin and origin != expected_origin):
            return self.send(dict(error='Invalid review token or origin'), status=403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 100_000:
                raise ValueError('Invalid request length')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/save':
                label = self.server.analysis.save(data['case_id'], data['annotation'], data['revision'])
                return self.send(dict(annotation=label, summary=self.server.analysis.summary()))
            if self.path == '/api/export':
                return self.send(self.server.analysis.export())
            return self.send(dict(error='Not found'), status=404)
        except Conflict as error:
            return self.send(dict(error=str(error)), status=409)
        except (ValueError, KeyError, TypeError) as error:
            return self.send(dict(error=str(error)), status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    analysis = Analysis(args.root)
    if not (analysis.output / 'manifest.json').exists():
        analysis.export()
    server = ReviewServer(('127.0.0.1', args.port), analysis)
    print(f'Unique error review: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
