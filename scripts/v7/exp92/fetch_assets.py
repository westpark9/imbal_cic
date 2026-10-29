"""Pinned encoder sources/weights and selective download of public CIC PCAPs."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import repo_root, resolve_path
from sota_sources import restore

ROOT = repo_root(__file__)
SOURCE = dict(name='netFound', url='https://github.com/SNL-UCSB/netFound.git',
              path='tabpfn/third_party/upstream/netFound', commit='b3ab5a3aa72640cc725ef207fb0145b039a57d35')
MODEL = 'snlucsb/netFound-small'
REVISION = 'e27221dfe97994ff880e892a32901fdd233817c5'
CIC_URL = ('https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/'
           'Original%20Network%20Traffic%20and%20Log%20data/Wednesday-14-02-2018/pcap.zip')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()

class HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP object; require byte ranges, pin ETag, cache bounded blocks."""
    def __init__(self, url, block_size=8 << 20):
        import requests
        self.url, self.session, self.pos, self.block_size = url, requests.Session(), 0, block_size
        r = self.session.head(url, timeout=45, allow_redirects=True)
        r.raise_for_status()
        self.size, self.etag = int(r.headers['Content-Length']), r.headers.get('ETag')
        self.cache_start, self.cache = -1, b''
        self.downloaded = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('Negative seek')
        return self.pos

    def read(self, size=-1):
        size = self.size-self.pos if size < 0 else min(size, self.size-self.pos)
        chunks = []
        while size > 0:
            if not self.cache_start <= self.pos < self.cache_start+len(self.cache):
                end = min(self.size-1, self.pos+self.block_size-1)
                headers = {'Range': f'bytes={self.pos}-{end}'}
                if self.etag: headers['If-Match'] = self.etag
                with self.session.get(self.url, headers=headers, stream=True, timeout=90) as r:
                    r.raise_for_status()
                    if r.status_code != 206: raise RuntimeError('Server did not honor Range; full archive download refused')
                    if r.headers.get('Content-Range') != f'bytes {self.pos}-{end}/{self.size}':
                        raise RuntimeError('Unexpected Content-Range')
                    self.cache = r.content
                if len(self.cache) != end-self.pos+1: raise IOError('Short HTTP range')
                self.cache_start, self.downloaded = self.pos, self.downloaded+len(self.cache)
            offset = self.pos-self.cache_start
            part = self.cache[offset:offset+size]
            chunks.append(part); self.pos += len(part); size -= len(part)
        return b''.join(chunks)

    def close(self):
        self.session.close()
        super().close()

def download_member(url, member, dest, max_bytes=2 << 30):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_suffix(dest.suffix+'.source.json')
    if dest.exists():
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['url'] == url and saved['member'] == member and saved['sha256'] == sha256(dest):
                return saved
        raise FileExistsError(f'Existing file has no matching verified receipt: {dest}')
    partial = dest.with_suffix(dest.suffix+'.partial')
    if partial.exists(): raise FileExistsError(f'Review interrupted file before retry: {partial}')
    with HTTPRangeFile(url) as remote, zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(member)
        if max(info.file_size, info.compress_size) > max_bytes:
            raise ValueError(f'Member exceeds download limit: {info.file_size} bytes')
        with archive.open(info) as src, partial.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        if partial.stat().st_size != info.file_size: raise IOError('Incomplete ZIP member')
        saved = dict(url=url, member=member, bytes=info.file_size, zip_crc32=info.CRC,
                     archive_etag=remote.etag, downloaded_bytes=remote.downloaded, sha256=sha256(partial))
        partial.rename(dest)
    receipt.write_text(json.dumps(saved, indent=2)+'\n')
    return saved

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['encoder', 'cic2018-pilot', 'zip-member'])
    p.add_argument('--url'); p.add_argument('--member'); p.add_argument('--output')
    args = p.parse_args()
    if args.action == 'encoder':
        if subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', SOURCE['path']+'/.git/HEAD']).returncode:
            raise RuntimeError('Encoder clone must be gitignored')
        state = restore(ROOT, SOURCE)
        if state['status'] != 'ok': raise RuntimeError(state)
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL, revision=REVISION, local_dir=ROOT/'data/pcap_models/netfound-small',
                                 allow_patterns=['config.json','model.safetensors','README.md'])
        print(json.dumps(dict(source=state, checkpoint=path, revision=REVISION)))
    else:
        url = args.url or CIC_URL
        member = args.member or 'pcap/UCAP172.31.69.25'
        output = args.output or 'data/pcap/cic2018/20180214_victim.pcap'
        print(json.dumps(download_member(url, member, resolve_path(output)), indent=2))

if __name__ == '__main__': main()
