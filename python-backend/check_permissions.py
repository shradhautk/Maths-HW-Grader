"""Checks each Python Lambda's IAM services (lambdas.py → deploy/template.yaml) against
what its code actually reaches, using the graphify graph of the repo.

For each function the graph is walked from functions/<name>.py into gradeant_common
(the only other code in its zip + layers); every reached file is scanned for the AWS
services it calls (boto3 clients/resources, Claude on Bedrock). A service the code
reaches but the function doesn't declare would fail at runtime with AccessDenied.

    python check_permissions.py                  # ../extra/leftovers/graphify-out/graph.json
    python check_permissions.py path/graph.json

Refresh the graph after code changes (repo root): graphify update .
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from lambdas import FUNCTIONS  # noqa: E402

GRAPH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'extra', 'leftovers', 'graphify-out', 'graph.json')
BOTO = re.compile(r"boto3\.(?:client|resource)\(\s*['\"]([\w-]+)['\"]")
SERVICE = {'dynamodb': 'dynamodb', 's3': 's3', 'ses': 'ses', 'sns': 'sns', 'lambda': 'lambda', 'secretsmanager': 'secrets'}
FOLLOW = {'imports', 'imports_from', 'calls', 'indirect_call', 'dynamic_import'}
PREFIX = 'python-backend/'


def load_graph():
    if not os.path.isfile(GRAPH):
        raise SystemExit(f'Graph not found: {GRAPH}\nBuild it from the repo root with: graphify update .')
    with open(GRAPH, encoding='utf-8') as fh:
        g = json.load(fh)
    file_of = {n['id']: (n.get('source_file') or '').replace('\\', '/') for n in g['nodes']}
    deps = {}
    for e in g['links']:
        if e.get('relation') not in FOLLOW:
            continue
        src, dst = file_of.get(e['source']), file_of.get(e['target'])
        if src and dst and src != dst and src.startswith(PREFIX) and dst.startswith(PREFIX):
            deps.setdefault(src, set()).add(dst)
    return g, set(file_of.values()), deps


def reach(start, deps):
    """Files reachable from `start`, staying inside the function's own file + gradeant_common."""
    seen, stack, cross = set(), [start], set()
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        for d in deps.get(f, ()):
            if d.startswith(PREFIX + 'gradeant_common/'):
                stack.append(d)
            elif d != start and d.startswith(PREFIX + 'functions/'):
                cross.add(d)  # another function's file: not in this zip, so not reachable at runtime
    return seen, cross


def services_in(path):
    with open(os.path.join(REPO, path), encoding='utf-8') as fh:
        src = fh.read()
    found = {SERVICE[s] for s in BOTO.findall(src) if s in SERVICE}
    if 'AnthropicBedrockMantle' in src:
        found.add('bedrock')
    return found


def main():
    g, files, deps = load_graph()
    built = g.get('built_at_commit')
    print(f"Checking Python Lambda permissions against {os.path.relpath(GRAPH)}" + (f' (graph at commit {str(built)[:8]})' if built else '') + '\n')
    problems = 0
    for name, spec in FUNCTIONS.items():
        entry = f'{PREFIX}functions/{name}.py'
        if entry not in files:
            print(f'  STALE gradeant-{name}: {entry} is not in the graph; run `graphify update .` from the repo root')
            problems += 1
            continue
        reached, cross = reach(entry, deps)
        used = set().union(*(services_in(f) for f in reached))
        missing = sorted(used - set(spec['aws']))
        extra = sorted(set(spec['aws']) - used)
        problems += bool(missing)
        print(f"  {'FAIL' if missing else 'ok  '} gradeant-{name:<12} {len(reached):>2} files  reaches: {', '.join(sorted(used)) or '—'}")
        if missing:
            print(f'         missing from lambdas.py aws (and its IAM role): {", ".join(missing)}')
        if extra:
            print(f'         declared but not reached: {", ".join(extra)} (harmless; can be removed)')
        if cross:
            print(f'         note: graph links to {", ".join(sorted(c[len(PREFIX):] for c in cross))} '
                  '(name match only; another function’s file is not in this zip)')
    if problems:
        raise SystemExit(f'\n{problems} function(s) need attention.')
    print('\nEvery function’s IAM role covers the AWS services its code reaches.')


if __name__ == '__main__':
    main()
