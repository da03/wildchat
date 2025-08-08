#!/usr/bin/env python3
import os
import re
import glob
import json
import tempfile
import subprocess
from collections import Counter
import argparse

import torch
from tqdm import tqdm


def sanitize_detector_name(name: str) -> str:
    """Make detector name safe for placeholder tags."""
    tag = re.sub(r'[^A-Z0-9]+', '_', name.upper()).strip('_')
    return tag[:64]


def write_chunk_to_tempfile(d):
    """Write all turns from the loaded chunk into a temp file."""
    total_turns = sum(len(conv) for conv in d.get('conversation', []))
    tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
    with tmp as f, tqdm(total=total_turns, desc='Building scan input', unit='turn') as pbar:
        for conv in d['conversation']:
            for turn in conv:
                content = turn.get('content')
                if isinstance(content, str) and content:
                    f.write(content)
                    f.write('\n')
                pbar.update(1)
    return tmp.name, total_turns


def run_trufflehog_on_file(tmp_filename, trufflehog_bin='trufflehog', results='verified'):
    """Run trufflehog on a text file and return {raw_secret: {'type':..., 'verified':...}}."""
    cmd = [trufflehog_bin, 'filesystem', '--json', f'--results={results}', tmp_filename]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    findings = {}
    type_counts = Counter()

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw = obj.get('Raw')
        det = obj.get('DetectorName') or 'SECRET'
        ver = bool(obj.get('Verified', False))
        if not raw:
            continue

        if raw in findings:
            if ver and not findings[raw]['verified']:
                findings[raw] = {'type': det, 'verified': ver}
        else:
            findings[raw] = {'type': det, 'verified': ver}

    for v in findings.values():
        type_counts[v['type']] += 1

    if proc.stderr:
        print('[trufflehog stderr]', proc.stderr.strip())

    return findings, type_counts


def main(save_name, trufflehog_bin='trufflehog', results='verified'):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')]

    files = glob.glob(
        f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.redacted.chunk*.pt'
    )

    def chunk_index(path):
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None

    files = sorted(files, key=chunk_index)

    for chunk_idx, file in enumerate(tqdm(files, desc='Chunks', unit='file')):
        print(f'\nLoading {file}')
        d = torch.load(file, weights_only=False)

        # Step 1: Build scan input
        tmp_path, total_turns = write_chunk_to_tempfile(d)

        # Step 2: Run TruffleHog once for the chunk
        print('Scanning with trufflehog...')
        secrets_map, by_type = run_trufflehog_on_file(tmp_path, trufflehog_bin=trufflehog_bin, results=results)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if not secrets_map:
            print('No secrets found.')
        else:
            print(f'Found {len(secrets_map)} unique secrets.')
            if by_type:
                top = ", ".join(f"{k}: {v}" for k, v in by_type.most_common(8))
                print(f'By detector type (top): {top}')

            # Step 3: Redact secrets with type-specific placeholders
            repl = {
                raw: f"<TRUFFLEHOG_REDACTED_{sanitize_detector_name(meta['type'])}>"
                for raw, meta in secrets_map.items()
            }

            with tqdm(total=total_turns, desc='Redacting', unit='turn') as pbar:
                for conv in d['conversation']:
                    for turn in conv:
                        content = turn.get('content')
                        orig = content
                        for raw, placeholder in repl.items():
                            if raw in content:
                                content = content.replace(raw, placeholder)
                        if content != orig:
                            turn['content'] = content
                            turn['redacted'] = True
                        pbar.update(1)

        # Step 4: Save with fixed naming convention
        out_file = f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.redacted.trufflehog.redacted.chunk{chunk_idx}.pt'
        torch.save(d, out_file)
        print(f'Saved: {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TruffleHog redaction on dataset chunks.')
    parser.add_argument('--save_name', type=str, required=True, help='Base name of the file to process (with or without .pt extension)')
    parser.add_argument('--trufflehog-bin', default='trufflehog', help='Path to trufflehog binary')
    parser.add_argument('--results', default='verified', help='TruffleHog results filter')
    args = parser.parse_args()

    main(args.save_name, trufflehog_bin=args.trufflehog_bin, results=args.results)
