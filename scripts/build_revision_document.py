#!/usr/bin/env python3
"""Compile author-supplied LaTeX in a temporary build directory, not Git history.

This is a document build utility, not an experiment. Temporary URLs are supplied
through workflow inputs, never stored in repository files or echoed into logs.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'document_build'
OUT=BUILD/'output'

def fetch(url: str, limit: int=4_000_000) -> bytes:
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Only HTTPS source URLs are permitted.')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'XMAG-Document-Build/1.0'}),timeout=90) as r:
        data=r.read(limit+1)
    if len(data)>limit: raise ValueError('Document exceeds build input size limit.')
    return data

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=fetch(os.environ['SOURCE_URL'])
    text=raw.decode('utf-8-sig').replace('\r\n','\n')
    if '\\begin{document}' not in text or '\\end{document}' not in text:
        raise ValueError('Input is not a complete LaTeX document.')
    # A complete owner-authored source is supplied as data, not executed as shell.
    source=BUILD/'XMAG_COS_Round2_Revised.tex'
    source.write_text(text,encoding='utf-8')
    hist=BUILD/'round2_histograms.tex'
    subprocess.run([sys.executable,str(ROOT/'scripts/render_round2_histogram_tex.py'),'--output',str(hist)],check=True)
    anchor='\\section{Attribution-Proxy Fidelity and Computational Scope}'
    if '\\input{round2_histograms.tex}' not in text:
        assert text.count(anchor)==1
        text=text.replace(anchor,'\\input{round2_histograms.tex}\n\n'+anchor)
        source.write_text(text,encoding='utf-8')
    # Plain article fallback compiles when an MDPI Definitions directory is absent.
    compile_log=[]
    for pass_number in range(1,4):
        result=subprocess.run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',source.name],cwd=BUILD,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        compile_log.append(f'PASS {pass_number}\n'+result.stdout)
        (OUT/'compile_log.txt').write_text('\n'.join(compile_log),encoding='utf-8')
        if result.returncode:
            print(result.stdout[-18000:])
            raise RuntimeError(f'LaTeX pass {pass_number} failed; retained complete log.')
    pdf=source.with_suffix('.pdf')
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    (OUT/'pdf_info.txt').write_text(info)
    subprocess.run(['pdftotext','-layout',str(pdf),str(OUT/'manuscript_text.txt')],check=True)
    final_log=source.with_suffix('.log').read_text(errors='replace')
    unresolved=('There were undefined references' in final_log or bool(re.search(r'(Citation|Reference) .+ undefined',final_log)))
    # Include the complete standalone source, histogram data, and optional reply.
    shutil.copy2(pdf,OUT/pdf.name)
    shutil.copy2(source,OUT/source.name)
    shutil.copy2(hist,OUT/hist.name)
    for env,name in [('RESPONSE_PDF_URL','Response_to_Reviewer_Round2.pdf'),('RESPONSE_TEX_URL','Response_to_Reviewer_Round2.tex')]:
        url=os.environ.get(env,'')
        if url:
            value=fetch(url)
            if name.endswith('.pdf') and not value.startswith(b'%PDF'):raise ValueError('Response download is not PDF.')
            (OUT/name).write_bytes(value)
    manifest={'build_status':'compiled' if not unresolved else 'compiled_with_unresolved_references','source_download_sha256':hashlib.sha256(raw).hexdigest(),'compiled_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'page_info':info,'undefined_references':unresolved,'overfull_box_count':final_log.count('Overfull \\hbox'),'format':'standalone article fallback; MDPI class used only when author supplies Definitions/mdpi.cls','scientific_scope':'Typesets supplied manuscript; does not train models, alter measured results, or certify empirical assumptions.','visual_review':'Not performed by this script.'}
    (OUT/'BUILD_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'README.txt').write_text('Complete revised manuscript and reviewer response. Compile XMAG_COS_Round2_Revised.tex with round2_histograms.tex in the same directory. Add the submitted MDPI Definitions folder to select the publisher class instead of the standalone fallback. No manuscript source was committed to Git. BUILD_MANIFEST.json states the executed checks and visual-review limitation.\n')
    print(json.dumps(manifest,indent=2))
    if unresolved:raise RuntimeError('Unresolved references require correction.')

if __name__=='__main__':main()
