"""Unified entry point for the DrugPack source release."""
import argparse
import json
import subprocess

from . import __version__
from .runtime import child_environment, project_root, python_executable

MODULES = {
    'baseline': 'evaluation.evaluate',
    'grozi': 'evaluation.grozi',
    'milk': 'evaluation.milk10k',
    'forb': 'evaluation.forb',
    'foreground': 'evaluation.foreground',
    'fusion': 'evaluation.bm25_image_fusion',
    'layout-fusion': 'evaluation.bm25_layout_fusion',
    'relative-layout-fusion': 'evaluation.bm25_relative_layout_fusion',
    'cache': 'evaluation.populate_cache',
    'cache-scores': 'evaluation.complete_scores',
}
METHOD = ('prepare', 'hidden', 'roma', 'features', 'heads', 'post', 'reader',
         'scalar_head', 'select', 'evaluate', 'baselines', 'fusion', 'benchmark',
         'queue', 'dispatch', 'report', 'figures', 'verify')
VERIFY = {
    'dataset': 'scripts/verify_release.py',
    'results': 'scripts/verify_results.py',
    'grozi': 'scripts/verify_grozi.py',
    'milk': 'scripts/verify_milk10k.py',
    'forb': 'scripts/verify_forb.py',
    'images': 'scripts/verify_image_embeddings.py',
    'text': 'scripts/verify_ocr_embeddings.py',
}
COMMANDS = (*MODULES, 'method', 'verify', 'ocr', 'review', 'review-errors', 'analyze-errors', 'doctor', 'build-release')


def workflow(command, arguments, root):
    """Translate release commands to existing entry points without changing options."""
    extra = list(arguments)
    if extra[:1] == ['--']:
        extra.pop(0)
    role = 'main'
    if command in MODULES:
        target = ['-m', MODULES[command]]
        if command == 'milk' and ('--model=biomedclip' in extra or
                any(a == '--model' and b == 'biomedclip' for a, b in zip(extra, extra[1:]))):
            role = 'biomed'
    elif command in ('method', 'verify', 'ocr'):
        choices = (METHOD if command == 'method' else tuple(VERIFY) + ('method',)
                   if command == 'verify' else ('drugpack', 'grozi', 'forb'))
        p = argparse.ArgumentParser(prog=f'drugpack {command}')
        p.add_argument('target', choices=choices)
        p.add_argument('arguments', nargs=argparse.REMAINDER)
        args = p.parse_args(extra)
        extra = args.arguments
        if extra[:1] == ['--']:
            extra.pop(0)
        if command == 'method':
            if args.target in ('dispatch', 'report', 'figures'):
                # These legacy modules have no parser; do not execute on --help.
                argparse.ArgumentParser(prog=f'drugpack method {args.target}',
                    description='Generate the saved experiment output; no command options.').parse_args(extra)
            target = ['-m', 'drugpack.method.' + args.target]
        elif command == 'verify':
            target = (['-m', 'drugpack.method.verify'] if args.target == 'method'
                      else [str(root / VERIFY[args.target])])
        else:
            role = 'ocr'
            target = ['-m', {'drugpack': 'evaluation.ocr', 'grozi': 'evaluation.grozi_ocr',
                              'forb': 'evaluation.forb_ocr'}[args.target]]
    elif command in ('review-errors', 'analyze-errors'):
        target = [str(root / ('dataset/tools/review_errors.py' if command == 'review-errors'
                              else 'dataset/tools/error_analysis.py'))]
    else:
        target = [str(root / 'dataset/tools/review_photos.py')]
    return [python_executable(root, role), '-B', *target, *extra]


def main(argv=None):
    p = argparse.ArgumentParser(prog='drugpack', description=__doc__, epilog=(
        'Global options precede the command. Use COMMAND --help for workflow options. '
        'doctor is read-only; baseline/verify/method may write their documented outputs.'))
    p.add_argument('--version', action='version', version=f'DrugPack {__version__}')
    p.add_argument('--root', help='Source/archive root (default: DRUGPACK_ROOT or this checkout)')
    p.add_argument('--dry-run', action='store_true', help='Print a workflow command without starting it')
    p.add_argument('command', choices=COMMANDS)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    try:
        root = project_root(args.root)
        if args.command == 'doctor':
            from .doctor import inspect
            q = argparse.ArgumentParser(prog='drugpack doctor', description='Read-only inventory; no model loading.')
            q.add_argument('--profile', choices=('source', 'baseline', 'method'), default='source')
            q.add_argument('--json', action='store_true')
            options = q.parse_args(args.arguments)
            report = inspect(root, options.profile)
            if options.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(f'DrugPack {__version__} | {root} | {options.profile}')
                for check in report['checks']:
                    print(f"{check['status'].upper():7} {check['name']}: {check['detail']}")
            return 0 if report['ok'] else 1
        if args.command == 'build-release':
            from .release import build
            q = argparse.ArgumentParser(prog='drugpack build-release', description='Code archive; excludes datasets, weights and caches.')
            q.add_argument('--output', required=True)
            options = q.parse_args(args.arguments)
            if args.dry_run:
                p.error('--dry-run applies to workflows, not build-release')
            print(json.dumps(build(root, options.output), indent=2))
            return 0
        command = workflow(args.command, args.arguments, root)
        if args.dry_run:
            print(json.dumps(dict(command=command, cwd=str(root)), ensure_ascii=False, indent=2))
            return 0
        return subprocess.run(command, cwd=root, env=child_environment(root), check=False).returncode
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
