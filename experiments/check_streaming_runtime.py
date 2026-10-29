"""Real ONNX + local TCP checks: eight calls per chunk, cancellation, one worker."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import Engine, SynthesisCancelled
import argparse
import io
import json
import socket
import threading
import time
import tempfile
import httpx
import numpy as np
import soundfile as sf
import uvicorn

from server import create_app
from streaming import split_text


class CountedSession:
    def __init__(self,session):
        self.session=session
        self.calls=0
        self.threads=set()

    def __getattr__(self,name):
        return getattr(self.session,name)

    def run(self,*args,**kwargs):
        self.calls+=1
        self.threads.add(threading.current_thread().name)
        return self.session.run(*args,**kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output')
    e=Engine(model=a.model)
    counter=CountedSession(e.sessions['vector_estimator']);e.sessions['vector_estimator']=counter
    stop=threading.Event();stop.set()
    try:
        e.synthesize('A cancelled sentence.',lang='en',cancel_event=stop)
        raise AssertionError('Expected cancellation')
    except SynthesisCancelled:
        assert counter.calls==0
    report=dict(model=Path(a.model).name,scope='Real ONNX on Mac CPU; local TCP, no ARM inference.',
                pre_cancelled_vector_calls=counter.calls,rows=[],complete=False)
    with tempfile.TemporaryDirectory() as custom:
        app=create_app(engine=e,custom_styles_dir=custom)
        config=uvicorn.Config(app,host='127.0.0.1',port=0,log_level='error',access_log=False)
        sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(16)
        url='http://127.0.0.1:'+str(sock.getsockname()[1])
        server=uvicorn.Server(config)
        thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True)
        thread.start()
        try:
            deadline=time.monotonic()+30
            while not server.started:
                if time.monotonic()>deadline:raise TimeoutError('Server startup')
                time.sleep(0.01)
            with httpx.Client(timeout=30) as client:
                for voice in ['M1','F1']:
                    for speed in [1.0,2.0]:
                        text='첫 번째 문장을 읽습니다. 두 번째 문장을 준비합니다. 마지막 문장도 빠짐없이 읽습니다.'
                        payload=dict(text=text,lang='ko',voice=voice,speed=speed,steps=8,
                                     first_chunk_length=24,max_chunk_length=28,silence_duration=0)
                        chunks=split_text(text,'ko',28,24)
                        before=counter.calls
                        full=client.post(url+'/v1/tts',json=payload);full.raise_for_status()
                        calls=counter.calls-before
                        assert calls==len(chunks)*8,(calls,len(chunks))
                        before=counter.calls
                        stream=client.post(url+'/v1/tts/stream',json=payload);stream.raise_for_status()
                        stream_calls=counter.calls-before
                        assert stream_calls==len(chunks)*8
                        wave,rate=sf.read(io.BytesIO(full.content),dtype='int16')
                        np.testing.assert_array_equal(wave,np.frombuffer(stream.content,'<i2'))
                        report['rows'].append(dict(voice=voice,speed=speed,chunks=len(chunks),
                            complete_vector_calls=calls,stream_vector_calls=stream_calls,pcm16_bitidentical=True))
                # Close a real streaming connection after its first audio event.
                # Later ONNX work may already be running; it stops at an operation boundary.
                text='첫 문장을 읽습니다. 다음 문장을 준비합니다. '+('그다음 문장도 정확하게 읽습니다. '*50)
                payload=dict(text=text,lang='ko',max_chunk_length=40,first_chunk_length=20,
                             stream=True,stream_format='sse',steps=8)
                before=counter.calls
                with client.stream('POST',url+'/v1/tts',json=payload) as response:
                    response.raise_for_status()
                    kind=None
                    for line in response.iter_lines():
                        if line.startswith('event: '):kind=line[7:]
                        if kind=='audio' and line.startswith('data: '):break
                closed=time.perf_counter()
                deadline=time.monotonic()+5
                while True:
                    h=client.get(url+'/v1/health');h.raise_for_status()
                    if h.json()['admitted_requests']==0:break
                    if time.monotonic()>deadline:raise TimeoutError('Cancellation did not release admission')
                    time.sleep(0.02)
                recovery=client.post(url+'/v1/tts',json={'text':'취소 후에도 정상입니다.','lang':'ko'})
                recovery.raise_for_status()
                calls=counter.calls-before-8
                full_calls=len(split_text(text,'ko',40,20))*8
                assert calls<full_calls,(calls,full_calls)
                report['cancellation']=dict(aborted_vector_calls=calls,full_request_vector_calls=full_calls,
                    recovery_status=recovery.status_code,recovered_seconds=time.perf_counter()-closed,
                    admission_released=True)
                assert client.get(url+'/v1/health').json()['admitted_requests']==0
                assert counter.threads=={'tts_0'},counter.threads
                assert all(s.get_session_options().intra_op_num_threads==1 and
                           s.get_session_options().inter_op_num_threads==1 for s in e.sessions.values())
                report['onnx_calling_threads']=sorted(counter.threads)
                report['ort_intra_inter_threads']=[1,1]
        finally:
            server.should_exit=True;thread.join(10);sock.close()
            assert not thread.is_alive()
    report['complete']=True
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
