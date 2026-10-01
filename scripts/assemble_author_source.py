#!/usr/bin/env python3
"""Assemble an author-supplied LaTeX revision without committing the document.

The versioned files contain only this generic utility, workflow configuration,
and source URLs. Publication content is fetched into an ephemeral build folder
and delivered as an Actions artifact. This script never changes measurements.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

LIMIT = 2 * 1024 * 1024
BLOCK = re.compile(r'^%<XMAG_BLOCK:([A-Za-z0-9_]+)>\s*\n(.*?)^%</XMAG_BLOCK:\1>\s*$', re.M | re.S)
PLAN = re.compile(r'^%<XMAG_EDIT_PLAN>\s*\n(.*?)^%</XMAG_EDIT_PLAN>\s*$', re.M | re.S)


def fetch(url: str) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'files.strivemath.org':
        raise ValueError('Only the supplied HTTPS document-store origin is allowed.')
    request = urllib.request.Request(url, headers={'User-Agent': 'author-source-assembly/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        if urllib.parse.urlsplit(response.geturl()).hostname != 'files.strivemath.org':
            raise ValueError('Unexpected redirect origin.')
        data = response.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError('Document exceeded the permitted source size.')
    return data


def decode_block(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.strip():
            lines.append('')
        elif line.startswith('%|'):
            lines.append(line[3:] if line.startswith('%| ') else line[2:])
        else:
            raise ValueError('An edit block contains an unmarked source line.')
    return '\n'.join(lines).rstrip() + '\n'


def unique(text: str, needle: str) -> int:
    if not needle or text.count(needle) != 1:
        raise ValueError(f'Expected one exact edit anchor, found {text.count(needle)}: {needle!r}')
    return text.index(needle)


def strip_comments(text: str) -> str:
    result = []
    for line in text.splitlines():
        cut = len(line)
        for i, char in enumerate(line):
            if char != '%':
                continue
            preceding = 0
            j = i - 1
            while j >= 0 and line[j] == '\\':
                preceding += 1
                j -= 1
            if preceding % 2 == 0:
                cut = i
                break
        result.append(line[:cut])
    return '\n'.join(result)


def command_argument(text: str, command: str) -> str:
    marker = command + '{'
    start = unique(text, marker) + len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{' and (i == 0 or text[i-1] != '\\'):
            depth += 1
        elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
            depth -= 1
            if depth == 0:
                return text[start:i]
    raise ValueError('Unclosed metadata argument.')


def check_source(text: str) -> dict:
    clean = strip_comments(text)
    if clean.count('\\begin{document}') != 1 or clean.count('\\end{document}') != 1:
        raise ValueError('Expected one complete document.')
    depth = 0
    for m in re.finditer(r'(?<!\\)[{}]', clean):
        depth += 1 if m.group() == '{' else -1
        if depth < 0:
            raise ValueError('Unbalanced closing brace.')
    if depth:
        raise ValueError(f'Unbalanced braces: depth={depth}')
    stack = []
    for m in re.finditer(r'\\(begin|end)\{([^{}]+)\}', clean):
        if m.group(1) == 'begin':
            stack.append(m.group(2))
        elif not stack or stack.pop() != m.group(2):
            raise ValueError(f'Mismatched environment: {m.group(0)}')
    if stack:
        raise ValueError(f'Unclosed environments: {stack}')
    labels = re.findall(r'\\label\{([^{}]+)\}', clean)
    duplicates = [k for k,v in Counter(labels).items() if v != 1]
    if duplicates:
        raise ValueError(f'Duplicate labels: {duplicates}')
    refs = re.findall(r'\\(?:ref|eqref|pageref)\{([^{}]+)\}', clean)
    unresolved = sorted(set(refs) - set(labels))
    if unresolved:
        raise ValueError(f'Unresolved source references: {unresolved}')
    bib = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^{}]+)\}', clean)
    cites = {k.strip() for block in re.findall(r'\\cite(?:\[[^]]*\])?\{([^{}]+)\}', clean) for k in block.split(',')}
    if cites - set(bib):
        raise ValueError(f'Unresolved citations: {sorted(cites-set(bib))}')
    return {'document_complete': True, 'brace_balance': True,
            'environment_balance': True, 'unique_labels': len(labels),
            'resolved_cross_references': len(refs), 'bibliography_entries': len(bib),
            'resolved_citation_keys': len(cites), 'lines': len(text.splitlines()),
            'table_environments': clean.count('\\begin{table}') + clean.count('\\begin{longtable}'),
            'figure_environments': clean.count('\\begin{figure}'),
            'compiled': False, 'visual_layout_checked': False}


def main() -> None:
    request_path = Path(sys.argv[1]) if len(sys.argv)>1 else Path('docs/source_assembly_request.json')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    original_bytes = fetch(request['base_source_url'])
    patch_bytes = fetch(request['revision_source_url'])
    original = original_bytes.decode('utf-8-sig').replace('\r\n','\n')
    patch = patch_bytes.decode('utf-8-sig').replace('\r\n','\n')
    blocks = {m.group(1): decode_block(m.group(2)) for m in BLOCK.finditer(patch)}
    plans = list(PLAN.finditer(patch))
    if len(plans) != 1 or not blocks:
        raise ValueError('The original revision source must retain its complete marked edit plan and blocks.')
    operations = json.loads(decode_block(plans[0].group(1)))
    revised, edits = original, []
    for op in operations:
        replacement = blocks[op['block']] if 'block' in op else op['new']
        kind = op['op']
        before = len(revised)
        if kind == 'replace_prefix':
            matches = list(re.finditer('^'+re.escape(op['prefix'])+r'.*$', revised, re.M))
            if len(matches) != 1:
                raise ValueError(f'Prefix is not unique: {op["prefix"]!r}')
            m = matches[0]
            revised = revised[:m.start()] + replacement.rstrip('\n') + revised[m.end():]
        elif kind == 'insert_before':
            at = unique(revised, op['anchor'])
            revised = revised[:at] + replacement + '\n' + revised[at:]
        elif kind == 'replace_between':
            first = unique(revised, op['start'])
            last = unique(revised, op['end'])
            if last <= first:
                raise ValueError('Reversed replacement interval.')
            revised = revised[:first] + replacement + '\n' + revised[last:]
        elif kind == 'replace_once':
            unique(revised, op['old'])
            revised = revised.replace(op['old'], replacement, 1)
        else:
            raise ValueError(f'Unsupported edit operator: {kind}')
        edits.append({'operation':kind, 'block':op.get('block'), 'before_chars':before,'after_chars':len(revised)})
    # Keep metadata consistent in both the MDPI and article fallback branches.
    declaration = command_argument(revised, '\\dataavailability')
    start_marker = '\\section*{Data Availability Statement}\n'
    end_marker = '\\section*{Conflicts of Interest}'
    a = unique(revised, start_marker) + len(start_marker)
    b = unique(revised, end_marker)
    if b <= a:
        raise ValueError('Invalid fallback data declaration.')
    revised = revised[:a] + declaration + '\n' + revised[b:]
    for prefix in ['\\Author{','\\AuthorNames{','\\address{','\\corres{',
                   '\\newcommand{\\orcidauthorA}', '\\newcommand{\\orcidauthorB}']:
        old = re.findall('^'+re.escape(prefix)+'.*$', original, re.M)
        new = re.findall('^'+re.escape(prefix)+'.*$', revised, re.M)
        if not old or old != new:
            raise ValueError(f'Author metadata changed: {prefix}')
    bib_marker = '\\begin{thebibliography}'
    if original[unique(original,bib_marker):] != revised[unique(revised,bib_marker):]:
        raise ValueError('Original bibliography or final back matter was altered.')
    old_checks = check_source(original)
    checks = check_source(revised)
    for count in ['table_environments','figure_environments']:
        if checks[count] < old_checks[count]:
            raise ValueError(f'Original content was lost: {count}')
    if len(revised) <= len(original):
        raise ValueError('This additive revision unexpectedly shortened the complete manuscript.')
    revised = '% Complete results-integrated manuscript; source assembled without committing publication content.\n' + revised
    response = BLOCK.sub('', patch)
    response = PLAN.sub('', response)
    # Independent response contains ordinary complete LaTeX only, not internal edits.
    response = re.sub(r'\n{3,}', '\n\n', response)
    response_checks = check_source(response)
    out = Path('document_build/output')
    out.mkdir(parents=True,exist_ok=True)
    (out/'main_round3_revised.tex').write_text(revised,encoding='utf-8')
    (out/'Response_to_Reviewer_Round3.tex').write_text(response,encoding='utf-8')
    if request.get('response_pdf_url'):
        pdf = fetch(request['response_pdf_url'])
        if not pdf.startswith(b'%PDF'):
            raise ValueError('Reviewer-response download is not a PDF.')
        (out/'Response_to_Reviewer_Round3.pdf').write_bytes(pdf)
    report = {'source_assembly':'passed', 'edit_count':len(edits), 'edits':edits,
              'original_source_sha256':hashlib.sha256(original_bytes).hexdigest(),
              'revision_source_sha256':hashlib.sha256(patch_bytes).hexdigest(),
              'output_source_sha256':hashlib.sha256(revised.encode()).hexdigest(),
              'preserved_authorship':True,'preserved_original_bibliography':True,
              'manuscript_checks':checks,'response_checks':response_checks,
              'scientific_verification_scope':'The revision uses the author-supplied result tables. This assembly does not rerun models or independently reconstruct raw scores.',
              'publication_source_committed':False}
    (out/'SOURCE_CHECKS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'README.txt').write_text('Upload main_round3_revised.tex to the existing Overleaf project. Keep Definitions/ from the original MDPI project and select this file as Main document. Compile with pdfLaTeX. All original bibliography entries and figures remain embedded; no external plotting run or BibTeX run is required. The new results, precision theorems, and replacement comparisons are integrated into the complete manuscript. The response is an independent document. Structural source checks passed, but this job does not compile or visually validate the complete manuscript. Verify the sponsor-approved funding statement and actual award number before submission; no grant placeholder has been inserted.\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Complete source artifact: document_build/output/main_round3_revised.tex')


if __name__=='__main__':
    main()
