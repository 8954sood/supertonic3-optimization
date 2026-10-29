"""Measure real local HTTP time-to-first-audio, RTF and PCM equivalence."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import time
from urllib.parse import urlparse

import httpx
import numpy as np
import soundfile as sf

TEXTS = {
    'ko': '안녕하세요. 오래된 컴퓨터에서도 빠르게 음성을 생성합니다. 첫 문장을 듣는 동안 다음 문장을 준비합니다. 자연스러운 발음과 끊김 없는 재생을 함께 확인해 보세요.',
    'en': 'Hello. This voice is generated locally using a single CPU thread. You can hear the first sentence while the next one is being prepared. Listen for clear pronunciation and continuous playback.',
}


def receive(client, url, payload):
    started = time.perf_counter()
    pcm, parts, first_audio, cursor, underruns, end = [], [], None, None, [], None
    with client.stream('POST',url+'/v1/tts',json={**payload,'stream':True,'stream_format':'sse'}) as response:
        response.raise_for_status()
        server_first = float(response.headers['X-First-Audio-Latency'])
        kind, lines = None, []
        for line in response.iter_lines():
            if line.startswith('event: '):
                kind = line[7:]
            elif line.startswith('data: '):
                lines.append(line[6:])
            elif not line and lines:
                data = json.loads('\n'.join(lines)); lines = []
                now = time.perf_counter()-started
                if kind == 'start':
                    sample_rate = data['sample_rate']
                elif kind == 'audio':
                    if first_audio is None:
                        first_audio, cursor = now, now+0.1
                    gap = max(0, now-cursor)
                    if gap:
                        underruns.append(gap)
                    cursor = max(cursor,now)+data['duration_s']
                    raw = base64.b64decode(data['audio_base64'])
                    assert len(raw) == data['samples']*2
                    pcm.append(raw)
                    parts.append(dict(index=data['index'],arrival_seconds=now,
                                      audio_seconds=data['duration_s'],generation_seconds=data['synthesis_seconds']))
                elif kind == 'end':
                    end = data
                elif kind == 'error':
                    raise RuntimeError(data)
    assert end and first_audio is not None and len(parts)>1
    elapsed = time.perf_counter()-started
    body = b''.join(pcm)
    assert len(body)/2/sample_rate == end['audio_seconds']
    return body, sample_rate, dict(first_audio_seconds=first_audio, server_first_audio_seconds=server_first,
        wall_seconds=elapsed, synthesis_seconds=end['synthesis_seconds'],audio_seconds=end['audio_seconds'],
        synthesis_rtf=end['rtf'],http_wall_rtf=elapsed/end['audio_seconds'],parts=parts,
        first_audio_precedes_completion=first_audio < elapsed,
        playback_simulation_initial_buffer_seconds=0.1,simulated_underrun_seconds=sum(underruns),
        note='Playback buffer simulation from complete SSE audio-event arrivals; not an audio-device underrun measurement.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:7788')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--first-chunk-length',type=int)
    a=p.parse_args()
    if urlparse(a.url).hostname not in ('127.0.0.1','localhost') or a.output.exists() or a.repeats<1:
        p.error('Use a local server, a new output and positive repeats')
    folder=Path('samples')/a.output.stem;folder.mkdir(parents=True,exist_ok=False)
    report=dict(machine=platform.platform(),cpu=platform.processor(),transport='local TCP HTTP',
        threads=1,steps_per_text_chunk=8,scope='Portable ORT CPU on this Mac; not x86 AVX2 v4 or Oracle ARM timing.',
        first_chunk_length=a.first_chunk_length,
        rows=[],complete=False)
    with httpx.Client(timeout=90) as client:
        report['health']=client.get(a.url+'/v1/health').json()
        for voice in ['M1','F1']:
            for lang,text in TEXTS.items():
                for speed in [1.0,2.0]:
                    case=f'{voice}_{lang}_{speed:g}'
                    payload=dict(text=text,voice=voice,lang=lang,steps=8,speed=speed,seed=42)
                    if a.first_chunk_length is not None:
                        payload['first_chunk_length']=a.first_chunk_length
                    warm=client.post(a.url+'/v1/tts',json=payload);warm.raise_for_status()
                    reference,rate=sf.read(io.BytesIO(warm.content),dtype='int16')
                    runs=[]
                    for _ in range(a.repeats):
                        raw,stream_rate,metrics=receive(client,a.url,payload)
                        assert rate==stream_rate
                        np.testing.assert_array_equal(reference,np.frombuffer(raw,'<i2'))
                        runs.append(metrics)
                    sf.write(folder/f'{case}.wav',reference,rate,subtype='PCM_16')
                    row=dict(case=case,voice=voice,lang=lang,speed=speed,text=text,runs=runs,
                        median_first_audio_seconds=statistics.median(r['first_audio_seconds'] for r in runs),
                        median_synthesis_rtf=statistics.median(r['synthesis_rtf'] for r in runs),
                        median_http_wall_rtf=statistics.median(r['http_wall_rtf'] for r in runs),
                        all_pcm16_identical=True,pcm_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                        audio_path=str(folder/f'{case}.wav'))
                    report['rows'].append(row);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                    print(case,'TTFA',round(row['median_first_audio_seconds'],3),'RTF',round(row['median_synthesis_rtf'],3),flush=True)
    report['complete']=True;a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
