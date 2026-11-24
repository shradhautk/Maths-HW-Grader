"""Builds the Python Lambda packages (nothing is uploaded). Python 3.10+, stdlib only.

    dist/layer-gradeant-python-deps.zip    Layer: requirements.txt for Linux arm64 / Python 3.13 → /opt/python
    dist/layer-gradeant-python-common.zip  Layer: gradeant_common + config/app.yaml             → /opt/python/gradeant_common
    dist/gradeant-<name>.zip               Function code: <name>.py at the root (handler <name>.lambda_handler)
    dist/manifest.json                     Routes, services, tables, sizes and SHA-256 of every zip

    python build.py              build
    python build.py --verify     build, then unpack the zips the way Lambda does and invoke
                                 every function with an API Gateway event (needs Python 3.13
                                 available as `py -3.13` or `python3.13`, else the current one)

Every zip entry gets Unix permissions (0644) and a fixed timestamp, so the same sources
always give the same SHA-256, on Windows, Linux or CI.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DIST = os.path.join(HERE, 'dist')
STAGE = os.path.join(DIST, 'stage')
sys.path.insert(0, HERE)
from lambdas import ARCHITECTURE, FUNCTIONS, RUNTIME  # noqa: E402

PY_VERSION = RUNTIME.replace('python', '')           # '3.13'
PLATFORM = 'manylinux2014_aarch64' if ARCHITECTURE == 'arm64' else 'manylinux2014_x86_64'
MAX_UNZIPPED = 250 * 1024 * 1024                     # Lambda: function + all layers, unzipped
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def mb(n):
    return f'{n / 1048576:.1f} MB'


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def files_under(root):
    out = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != '__pycache__')
        for n in sorted(names):
            if n.endswith(('.pyc', '.pyo')):
                continue
            full = os.path.join(d, n)
            if not os.path.islink(full):
                out.append(os.path.relpath(full, root).replace(os.sep, '/'))
    return out


def write_zip(zip_path, root, files, prefix=''):
    """Deterministic zip with Unix 0644 entries; returns unzipped bytes."""
    total = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            with open(os.path.join(root, rel), 'rb') as fh:
                data = fh.read()
            total += len(data)
            info = zipfile.ZipInfo(prefix + rel, date_time=FIXED_TIME)
            info.create_system = 3                      # Unix, so the mode below is honoured
            info.external_attr = (0o100644 << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return total


DEFAULT_LAYERS = ['deps', 'common']
LAYER_NAMES = {'deps': 'gradeant-python-deps', 'common': 'gradeant-python-common', 'pipeline-deps': 'gradeant-pipeline-deps'}


def pip_layer(manifest, layer_name, requirements, stage_dir):
    """pip-installs a requirements file for the Lambda platform into a layer zip; returns its unzipped bytes."""
    target = os.path.join(STAGE, stage_dir)
    print(f'Installing {requirements} for {PLATFORM} / Python {PY_VERSION} …')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                    '--no-compile', '--no-warn-conflicts', '--target', target, '--platform', PLATFORM, '--implementation', 'cp',
                    '--python-version', PY_VERSION, '--only-binary=:all:', '--upgrade',
                    '-r', os.path.join(HERE, requirements)], check=True)
    files = [f for f in files_under(target) if not f.startswith('bin/')]
    zip_path = os.path.join(DIST, f'layer-{layer_name}.zip')
    size = write_zip(zip_path, target, files, prefix='python/')
    manifest['layers'][layer_name] = {'zip': os.path.basename(zip_path), 'mountsAt': '/opt/python',
                                      'files': len(files), 'unzippedBytes': size, 'sha256': sha256(zip_path)}
    return size


def build():
    shutil.rmtree(DIST, ignore_errors=True)
    os.makedirs(STAGE)
    manifest = {'runtime': RUNTIME, 'architecture': ARCHITECTURE, 'layers': {}, 'functions': {}}

    # ---- Layer 1: third-party packages, built for the Lambda platform ----
    deps_bytes = pip_layer(manifest, 'gradeant-python-deps', 'requirements.txt', 'deps')
    layer_bytes = {'deps': deps_bytes}

    # ---- Layer 1b: grading-pipeline packages, only when a worker uses them ----
    if any('pipeline-deps' in spec.get('layers', []) for spec in FUNCTIONS.values()):
        layer_bytes['pipeline-deps'] = pip_layer(manifest, 'gradeant-pipeline-deps', 'pipeline-requirements.txt', 'pipeline')

    # ---- Layer 2: the shared code + config/app.yaml ----
    common = os.path.join(STAGE, 'common', 'gradeant_common')
    shutil.copytree(os.path.join(HERE, 'gradeant_common'), common, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    os.makedirs(os.path.join(common, 'config'))
    shutil.copy2(os.path.join(REPO, 'config', 'app.yaml'), os.path.join(common, 'config', 'app.yaml'))
    common_root = os.path.dirname(common)
    common_files = files_under(common_root)
    common_zip = os.path.join(DIST, 'layer-gradeant-python-common.zip')
    common_bytes = write_zip(common_zip, common_root, common_files, prefix='python/')
    manifest['layers']['gradeant-python-common'] = {'zip': os.path.basename(common_zip), 'mountsAt': '/opt/python/gradeant_common',
                                                    'files': len(common_files), 'unzippedBytes': common_bytes, 'sha256': sha256(common_zip)}
    layer_bytes['common'] = common_bytes

    # ---- One zip per function ----
    for name, spec in FUNCTIONS.items():
        src = os.path.join(HERE, 'functions')
        zip_path = os.path.join(DIST, f'gradeant-{name}.zip')
        fn_bytes = write_zip(zip_path, src, [f'{name}.py'])
        layers = spec.get('layers', DEFAULT_LAYERS)
        total = fn_bytes + sum(layer_bytes[x] for x in layers)
        if total > MAX_UNZIPPED:
            raise SystemExit(f'gradeant-{name}: {mb(total)} unzipped with its layers, over the 250 MB limit')
        manifest['functions'][f'gradeant-{name}'] = {
            'zip': os.path.basename(zip_path), 'handler': f'{name}.lambda_handler', 'kind': spec.get('kind', 'api'),
            'layers': [LAYER_NAMES[x] for x in layers], 'routes': spec['routes'],
            'aws': spec['aws'], 'tables': spec['tables'], 'memoryMB': spec['memory'], 'timeoutSeconds': spec['timeout'],
            'unzippedWithLayers': mb(total), 'sha256': sha256(zip_path),
        }

    with open(os.path.join(DIST, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    shutil.rmtree(STAGE, ignore_errors=True)

    print('\nLayers:')
    for layer in manifest['layers'].values():
        size = os.path.getsize(os.path.join(DIST, layer['zip']))
        print(f"  {layer['zip']:<36} {layer['files']:>6} files {mb(size):>9} zipped {mb(layer['unzippedBytes']):>9} unzipped  → {layer['mountsAt']}")
    print(f'Functions ({RUNTIME}, {ARCHITECTURE}):')
    for f in manifest['functions'].values():
        what = ', '.join(f['routes']) or 'worker'
        print(f"  {f['zip']:<36} {f['handler']:<28} {what:<34} {f['unzippedWithLayers']:>9} with {' + '.join(f['layers'])}")
    print(f'\nWritten to {os.path.relpath(DIST)}')
    return manifest


# ---- Verify: unpack like Lambda and invoke each function ------------------------------

VERIFY_SCRIPT = r'''
import json, sys
task, opt, name, event = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
sys.path[:0] = [task, opt]                 # Lambda: /var/task first, then /opt/python
if not sys.platform.startswith('linux'):
    # The layers hold Linux arm64 wheels. cryptography (google-auth, for Gemini) can't load here,
    # and PyJWT imports it when present: hide it, so PyJWT uses its pure-Python HS256 path (the
    # only algorithm GradeAnt uses). On Lambda / Linux the real package loads.
    class _HideLinuxOnly:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'cryptography' or fullname.startswith('cryptography.'):
                raise ModuleNotFoundError(fullname)
    sys.meta_path.insert(0, _HideLinuxOnly())
mod = __import__(name)
res = mod.lambda_handler(event, None)
status = res.get('statusCode') if 'statusCode' in res else ('batch-ok' if res.get('batchItemFailures') == [] else repr(res))
print(json.dumps({'status': status, 'python': sys.version.split()[0]}))
'''


def _python313():
    for cmd in (['py', '-3.13'], ['python3.13']):
        try:
            subprocess.run(cmd + ['-c', 'pass'], check=True, capture_output=True)
            return cmd
        except (OSError, subprocess.CalledProcessError):
            continue
    print('  (Python 3.13 not found: verifying with the current interpreter)')
    return [sys.executable]


def verify(manifest):
    tmp = tempfile.mkdtemp(prefix='gradeant-py-')
    for layer in manifest['layers'].values():
        with zipfile.ZipFile(os.path.join(DIST, layer['zip'])) as zf:
            bad = [i.filename for i in zf.infolist() if (i.external_attr >> 16) & 0o777 != 0o644]
            if bad:
                raise SystemExit(f"{layer['zip']}: unexpected modes on {bad[:3]}")
            if zf.testzip() is not None:
                raise SystemExit(f"{layer['zip']}: corrupt entry")
    python = _python313()
    env = {'PATH': os.environ.get('PATH', ''), 'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),
           'APP_ENV': 'production', 'JWT_SECRET': 'package-verification-only', 'CORS_ALLOWED_ORIGINS': 'https://example.invalid',
           'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x', 'PYTHONDONTWRITEBYTECODE': '1'}
    failed = 0
    print('\nVerifying each function from its zip + layers:')
    for fn in manifest['functions'].values():
        name = fn['handler'].split('.')[0]
        task = os.path.join(tmp, f'task-{name}')
        with zipfile.ZipFile(os.path.join(DIST, fn['zip'])) as zf:
            zf.extractall(task)
        # Like Lambda: only this function's layers, in order, in its own /opt.
        opt = os.path.join(tmp, f'opt-{name}')
        for layer_name in fn['layers']:
            with zipfile.ZipFile(os.path.join(DIST, manifest['layers'][layer_name]['zip'])) as zf:
                zf.extractall(opt)
        py_path = os.path.join(opt, 'python')
        if fn.get('kind') == 'worker':
            event, want = {'Records': []}, 'batch-ok'
        elif name == 'health':
            event, want = {'httpMethod': 'GET', 'path': '/health', 'headers': {}}, 200
        else:
            route = fn['routes'][0].split(' ', 1)[1].replace('/*', '')
            event, want = {'httpMethod': 'OPTIONS', 'path': route, 'headers': {'Origin': 'https://example.invalid'}}, 204
        r = subprocess.run(python + ['-c', VERIFY_SCRIPT, task, py_path, name, json.dumps(event)],
                           capture_output=True, text=True, env=env, timeout=120)
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            out = {}
        ok = out.get('status') == want
        failed += not ok
        what = f"{event['httpMethod']} {event['path']}" if 'httpMethod' in event else 'SQS {"Records": []}'
        print(f"  {'ok  ' if ok else 'FAIL'} gradeant-{name:<12} {what} → {out.get('status')}"
              f" (Python {out.get('python', '?')})" + ('' if ok else f'\n{r.stdout}\n{r.stderr}'))
    shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        raise SystemExit(f'{failed} function(s) failed verification')
    print('\nAll functions start from their zips + layers and answer correctly.')


if __name__ == '__main__':
    m = build()
    if '--verify' in sys.argv:
        verify(m)
