#!/usr/bin/env python3
"""Regenerate REPORT.pdf -- the print/reading version of REPORT.html.

REPORT.html is the authoritative report; REPORT.pdf is a *derived* artifact
and must be regenerated (and the change committed) whenever REPORT.html or
its figures change.

The script builds a print copy of REPORT.html (a real <thead> per table so
print repeats the header row across page breaks + a print-only stylesheet),
renders it with a headless Chromium-based browser, and writes REPORT.pdf
next to REPORT.html. Only the standard library is required; the browser is
auto-detected (override with --browser or the CHROME_PATH environment
variable).

Usage:
    python3 plots/build_report_pdf.py                 # write ../REPORT.pdf
    python3 plots/build_report_pdf.py --out /tmp/x.pdf
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, os.pardir))

BROWSER_CANDIDATES = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    'microsoft-edge', 'microsoft-edge-stable', 'chromium', 'chromium-browser',
    'google-chrome', 'google-chrome-stable',
]

PRINT_CSS = '''
<style media="print">
@page { size: A4; margin: 14mm 12mm 16mm 12mm; }
html, body { background: #fff; }
.wrap { max-width: none; margin: 0; padding: 0; }
body { line-height: 1.55; }
h2 { page-break-after: avoid; margin-top: 30px; }
h3 { page-break-after: avoid; }
p { orphans: 3; widows: 3; }
table { page-break-inside: auto; font-size: 12px !important; }
td, th { word-break: keep-all; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
img, .fig, .figcap { page-break-inside: avoid; }
.figcap { page-break-before: avoid; }
img.fig { page-break-after: avoid; }
pre { white-space: pre-wrap !important; overflow-wrap: anywhere; overflow-x: visible !important; }
</style>
</head>'''


def find_browser(explicit=None):
    for cand in ([explicit] if explicit else []) + \
                [os.environ.get('CHROME_PATH')] + BROWSER_CANDIDATES:
        if not cand:
            continue
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        elif shutil.which(cand):
            return shutil.which(cand)
    sys.exit('no Chromium-based browser found; pass --browser <path> '
             'or set CHROME_PATH')


def build_print_copy(src_html, dst_html):
    html = open(src_html, encoding='utf-8').read()

    def add_thead(m):
        block = m.group(0)
        tm = re.search(r'(<table[^>]*>)(.*?)</table>', block, re.S)
        open_tag, inner = tm.group(1), tm.group(2)
        trm = re.search(r'(<tr>.*?</tr>)', inner, re.S)
        first_tr = trm.group(1)
        rest = inner.replace(first_tr, '', 1)
        return (f'{open_tag}<thead>{first_tr}</thead>'
                f'<tbody>{rest}</tbody></table>')

    n_tables = len(re.findall(r'<table[^>]*>.*?</table>', html, re.S))
    html = re.sub(r'<table[^>]*>.*?</table>', add_thead, html, flags=re.S)
    assert html.count('</head>') == 1
    html = html.replace('</head>', PRINT_CSS)
    with open(dst_html, 'w', encoding='utf-8', newline='') as fh:
        fh.write(html)
    return n_tables


def page_count(pdf):
    data = open(pdf, 'rb').read()
    n = len(re.findall(rb'/Type\s*/Page[^s]', data))
    m = re.search(rb'/Count (\d+)', data)
    if m:
        n = max(n, int(m.group(1)))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--browser')
    ap.add_argument('--out', default=os.path.join(PKG, 'REPORT.pdf'))
    ap.add_argument('--keep-print-copy', action='store_true',
                    help='keep the generated print copy for inspection')
    args = ap.parse_args()

    src = os.path.join(PKG, 'REPORT.html')
    if not os.path.exists(src):
        sys.exit(f'REPORT.html not found at {src}')

    # the print copy must live in the package root so that the relative
    # figures/ and katex/ paths resolve in the browser
    tmp = os.path.join(PKG, '.report_print_copy.html')
    browser = find_browser(args.browser)
    print(f'browser: {browser}')
    try:
        n = build_print_copy(src, tmp)
        print(f'tables given thead: {n}')
        cmd = [browser, '--headless=new', '--disable-gpu',
               '--no-pdf-header-footer', '--print-to-pdf-no-header',
               '--virtual-time-budget=30000',
               f'--print-to-pdf={os.path.abspath(args.out)}',
               'file:///' + tmp.replace(os.sep, '/')]
        t0 = time.time()
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        print('rendered in %.1fs -> %s (%d pages, %.1f MB)'
              % (time.time() - t0, args.out, page_count(args.out),
                 os.path.getsize(args.out) / 1e6))
    finally:
        if not args.keep_print_copy and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == '__main__':
    main()
