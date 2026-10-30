"""Alternate real local HTTP streams between frozen model packages."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import Engine
import argparse
import hashlib
import json
import platform
import socket
import statistics
import tempfile
import threading
import time
import httpx
import uvicorn
from server import create_app
from experiments.benchmark_streaming import TEXTS, receive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=3)
    a=p.parse_args()
    if a.output.exists() or a.pairs<3:p.error('Use a new output and at least three pairs')
    report=dict(reference=a.reference,candidate=a.candidate,machine=platform.platform(),
        threads_per_inference=1,steps_per_chunk=8,transport='Local TCP, alternating complete requests; no concurrent inference.',
        note='Same text/chunk/seed/voice/speed and PCM required. Mac timings are not ARM timings.',
        rows=[],complete=False)
    servers,threads,sockets,urls=[],[],[],{}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for name,model in [('candidate',a.candidate),('reference',a.reference)]:
                app=create_app(model_path=model,custom_styles_dir=Path(tmp)/name)
                sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(16)
                server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=0,log_level='error',access_log=False))
                thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True)
                servers.append(server);threads.append(thread);sockets.append(sock)
                thread.start();deadline=time.monotonic()+30
                while not server.started:
                    if not thread.is_alive() or time.monotonic()>deadline:raise RuntimeError('Server startup failed')
                    time.sleep(.01)
                urls[name]='http://127.0.0.1:'+str(sock.getsockname()[1])
            with httpx.Client(timeout=90) as client:
                for voice in ['M1','F1']:
                    for lang,text in TEXTS.items():
                        for speed in [1.0,2.0]:
                            payload=dict(text=text,voice=voice,lang=lang,speed=speed,steps=8,seed=42)
                            expected=None
                            for name in urls:
                                raw,rate,_=receive(client,urls[name],payload)
                                if expected is None:expected=(raw,rate)
                                assert (raw,rate)==expected
                            runs={name:[] for name in ['reference','candidate']}
                            for pair in range(a.pairs):
                                names=['reference','candidate']
                                if pair%2:names.reverse()
                                for name in names:
                                    raw,rate,metrics=receive(client,urls[name],payload)
                                    assert (raw,rate)==expected
                                    runs[name].append(metrics)
                            fields={}
                            for metric in ['first_audio_seconds','synthesis_seconds','wall_seconds']:
                                values=[100*(1-c[metric]/r[metric]) for r,c in zip(runs['reference'],runs['candidate'])]
                                fields[metric]=dict(reference=statistics.median(r[metric] for r in runs['reference']),
                                    candidate=statistics.median(r[metric] for r in runs['candidate']),
                                    paired_reductions_percent=values,median_reduction_percent=statistics.median(values))
                            row=dict(voice=voice,lang=lang,speed=speed,runs=runs,summary=fields,
                                pcm16_bitidentical=True,pcm_sha256=hashlib.sha256(expected[0]).hexdigest())
                            report['rows'].append(row);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                            print(voice,lang,speed,'wall reduction %',round(fields['wall_seconds']['median_reduction_percent'],2),flush=True)
            report['complete']=True
        finally:
            for server in servers:server.should_exit=True
            for thread in threads:thread.join(10)
            for sock in sockets:sock.close()
            report['servers_stopped']=all(not thread.is_alive() for thread in threads)
            a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert report['servers_stopped']


if __name__=='__main__':main()
