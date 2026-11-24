"""Open Images per-photo credits (credits.json: image id -> [author, licence, landing URL]) -> one compact text file.
  python3 -I img/web/credits.py <credits.json> <out.txt>
The input is untrusted data: it is only parsed as JSON, and every value written is filtered first. Author names lose
control characters and the characters < > (so the file is harmless even if something shows it as HTML); a name that
is an e-mail address keeps only the part before the @ (this file is public); a name with nothing printable in it
becomes the account name. Links are kept only if they are plain http(s) URLs made of ordinary URL characters.
Photos are grouped by author and Flickr account, the numbers wrapped at about 100 columns:
  Author name | account | photo photo photo ...        ->  https://www.flickr.com/photos/<account>/<photo>
"""
import html, json, re, sys, unicodedata

# a Flickr photo page, with or without www. and /photos/, and with any tail (/in/photostream, ?rb=1, ...), which is dropped
FLICKR = re.compile(r'https?://(?:www\.)?flickr\.com/(?:photos/)?([A-Za-z0-9_@.-]{1,64})/([0-9]{1,20})(?:[/?#].*)?', re.S)
NOT_ACCOUNT = {'photos', 'people', 'groups', 'explore', 'search', 'services', 'x', 'gp'}
EMAIL = re.compile(r'([A-Za-z0-9._%+-]+)@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
WIDTH = 100
URL_OK = re.compile(r"https?://[A-Za-z0-9.-]{1,100}(/[A-Za-z0-9._~/?#@!$&'()*+,;=%:-]{0,300})?")
LICENCES = {'https://creativecommons.org/licenses/by/2.0/': 'CC BY 2.0'}
ID_OK = re.compile(r'[0-9a-f]{16}')


def clean(s, limit=100):
    """A name as plain text on one line: entities decoded, no control/format characters, no < > |."""
    if not isinstance(s, str): return ''
    s = html.unescape(s)
    s = ''.join(' ' if unicodedata.category(c)[0] in 'CZ' else c for c in s)       # controls, format marks, odd spaces
    s = s.replace('<', '(').replace('>', ')').replace('|', '/')
    s = EMAIL.sub(lambda m: m.group(1), s)                                         # never publish an e-mail address
    s = ' '.join(s.split())[:limit]
    return s


def main(src, out):
    data = json.load(open(src, encoding='utf-8'))
    assert isinstance(data, dict)
    groups, other, skipped, lic = {}, [], 0, set()
    for k, v in data.items():
        if not (isinstance(k, str) and ID_OK.fullmatch(k) and isinstance(v, list) and len(v) == 3 and all(isinstance(x, str) for x in v)):
            skipped += 1; continue
        name, licence, url = clean(v[0]), LICENCES.get(v[1]), v[2]
        if licence is None: skipped += 1; continue
        lic.add(licence)
        m = FLICKR.fullmatch(url)
        if m and m.group(1).lower() not in NOT_ACCOUNT: groups.setdefault((name or m.group(1), m.group(1)), []).append(int(m.group(2)))
        else: other.append((name or '(unknown)', url if URL_OK.fullmatch(url) and not EMAIL.search(url) else '(link not kept)', k))
    n = sum(len(g) for g in groups.values()) + len(other)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('﻿')                                                          # so a browser reads the names as UTF-8
        f.write(f"""PocketPix: credits for the Open Images photographs used in training
======================================================================
{n} photographs from Open Images V7 (https://storage.googleapis.com/openimages/web/index.html).
Every photograph listed here is licensed {', '.join(sorted(lic))} (https://creativecommons.org/licenses/by/2.0/).
The object boxes used to cut the close-up crops are by Google LLC, licensed CC BY 4.0.
The photographs were cropped, shrunk to 64x64 pixels and reduced to 256 colours; none is included in this download.
Author names are as recorded by Open Images. See CREDITS.txt for the rest of the training data.

Format, one entry per author and Flickr account (long entries continue on indented lines):
  author | account | photo numbers
and each photograph's page is https://www.flickr.com/photos/ACCOUNT/NUMBER
Where an author's recorded name was an e-mail address, only the part before the @ is given.

""")
        for (name, acct), ids in sorted(groups.items(), key=lambda kv: (kv[0][0].casefold(), kv[0][1])):
            line = f'{name} | {acct} |'
            for i in sorted(set(ids)):
                if len(line) + 1 + len(str(i)) > WIDTH and not line.endswith('|'): f.write(line + '\n'); line = '   '
                line += f' {i}'
            f.write(line + '\n')
        if other:
            f.write('\nPhotographs with a different kind of link (author | link | Open Images id):\n')
            for name, url, k in sorted(other, key=lambda t: (t[0].casefold(), t[2])): f.write(f'{name} | {url} | {k}\n')
    print(out, n, 'photos,', len(groups), 'author lines,', len(other), 'other links,', skipped, 'skipped')


if __name__ == '__main__':
    main(*sys.argv[1:3])
