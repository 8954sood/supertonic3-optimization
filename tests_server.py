"""Contract and resource tests for real HTTP handlers, without model weights."""
from runtime import SynthesisCancelled
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf
from supertonic.core import Style

from server import AudioStreamingResponse, Lease, create_app
from streaming import split_text


class FakeEngine:
    sample_rate = 44100

    def __init__(self):
        self.calls, self.active, self.max_active = [], 0, 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.hold, self.fail = False, False
        self.fail_at = None

    def synthesize(self, text, lang, steps, seed, speed, *, voice_style, cancel_event):
        self.calls.append((text, lang, steps, seed, speed, float(voice_style.dp[0])))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.entered.set()
            if self.hold:
                while not self.release.wait(0.005):
                    if cancel_event.is_set():
                        raise SynthesisCancelled()
            if self.fail or len(self.calls) == self.fail_at:
                raise RuntimeError("injected synthesis failure")
            wave = np.linspace(-0.6, 0.6, 441, dtype=np.float32) * float(voice_style.dp[0])/2
            return wave, {"seconds": 0.001, "generation_speed": speed}
        finally:
            self.active -= 1


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = FakeEngine()
        self.voices = {name: Style(np.array([v], np.float32), np.array([v], np.float32))
                       for name, v in [("M1", 1), ("F1", 2)]}
        self.app = create_app(engine=self.engine, voice_styles=self.voices,
                              custom_styles_dir=self.tmp.name, max_pending=0)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        self.engine.release.set()
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_routes_and_openapi(self):
        self.assertEqual(self.client.get('/v1/health').json()['threads'], 1)
        self.assertEqual(self.client.get('/v1/health').json()['steps'], 8)
        self.assertEqual(len(self.client.get('/v1/styles').json()['styles']), 2)
        self.assertEqual(self.client.get('/v1/models').json()['data'][0]['id'], 'supertonic-3')
        self.assertIn('/v1/tts/stream', self.client.get('/openapi.json').json()['paths'])

    def test_stream_pcm_matches_complete_wav_and_alias(self):
        payload = dict(text='First sentence. Second sentence. Third sentence.', lang='en',
                       max_chunk_length=20, first_chunk_length=20, silence_duration=0.1, voice='F1')
        full = self.client.post('/v1/tts', json=payload)
        self.assertEqual(full.status_code, 200)
        wave, rate = sf.read(io.BytesIO(full.content), dtype='int16')
        stream = self.client.post('/v1/tts/stream', json=payload)
        self.assertEqual(stream.status_code, 200)
        np.testing.assert_array_equal(np.frombuffer(stream.content, '<i2'), wave)
        self.assertEqual(int(stream.headers['X-Sample-Rate']), rate)
        self.assertEqual(int(stream.headers['X-Chunk-Count']), 3)
        alias = self.client.post('/v1/audio/speech', json={
            **{k:v for k,v in payload.items() if k != 'text'},
            'model':'supertonic-3', 'input':payload['text'], 'stream':True, 'response_format':'pcm'})
        self.assertEqual(alias.content, stream.content)
        self.assertTrue(all(call[2] == 8 for call in self.engine.calls))
        self.assertEqual(self.app.state.service.admitted, 0)

    def test_sse_completion_and_pcm(self):
        payload = dict(text='First sentence. Second sentence.', lang='en', max_chunk_length=20,
                       stream=True, stream_format='sse', silence_duration=0)
        response = self.client.post('/v1/tts', json=payload)
        messages = [block.split('\n') for block in response.text.strip().split('\n\n')]
        self.assertEqual([m[0] for m in messages], ['event: start','event: audio','event: audio','event: end'])
        chunks = [json.loads(m[1][6:]) for m in messages[1:-1]]
        pcm = b''.join(base64.b64decode(c['audio_base64']) for c in chunks)
        full = self.client.post('/v1/tts', json={**payload, 'stream':False, 'response_format':'pcm'})
        self.assertEqual(pcm, full.content)
        end = json.loads(messages[-1][1][6:])
        self.assertAlmostEqual(end['audio_seconds'], len(pcm)/2/44100)
        self.assertEqual(end['chunks'], 2)

    def test_invalid_requests_do_no_inference(self):
        for payload, status in [({'steps':7},422),({'steps':50},422),({'steps':8.5},422),
                                ({'voice':'../M1'},400),({'lang':'unsupported'},400),
                                ({'speed':0.5},422),({'speed':1.251},422),({'response_format':'mp3'},400),
                                ({'stream':True,'response_format':'wav'},400),({'text':'  '},400)]:
            with self.subTest(payload=payload):
                r = self.client.post('/v1/tts', json={'text':'hello', **payload})
                self.assertEqual(r.status_code, status, r.text)
                self.assertIn('error', r.json())
        self.assertEqual(self.engine.calls, [])
        r = self.client.post('/v1/audio/speech',json={'model':'wrong','input':'hello'})
        self.assertEqual(r.status_code,400)

    def test_fractional_speed_to_two_decimal_places(self):
        for value in [0.70,1.0,1.25,1.30,1.99,2.0]:
            r=self.client.post('/v1/tts',json={'text':'Hello','speed':value})
            self.assertEqual(r.status_code,200,r.text)
            self.assertEqual(self.engine.calls[-1][4],value)
        for path,payload in [('/v1/tts',{'text':'Hello','speed':1.255}),
                             ('/v1/audio/speech',{'input':'Hello','model':'supertonic-3','speed':1.255}),
                             ('/v1/tts/batch',{'items':[{'text':'Hello'}],'defaults':{'speed':1.255}})]:
            self.assertEqual(self.client.post(path,json=payload).status_code,422)

    def test_batch_defaults_and_formats(self):
        for fmt in ['wav','flac','ogg','pcm']:
            with self.subTest(fmt=fmt):
                r = self.client.post('/v1/tts/batch', json={'items':[{'text':'Hello'}, {'text':'Hi','voice':'M1'}],
                    'defaults':{'voice':'F1','speed':1.5}, 'response_format':fmt})
                self.assertEqual(r.status_code,200,r.text)
                self.assertEqual(len(r.json()['items']),2)
                self.assertEqual([c[-1] for c in self.engine.calls[-2:]], [2,1])
                self.assertTrue(base64.b64decode(r.json()['items'][0]['audio_base64']))
                self.assertEqual(self.app.state.service.admitted,0)

    def test_queue_limit_and_responsive_health(self):
        self.engine.hold = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.client.post,'/v1/tts',json={'text':'A first request'})
            self.assertTrue(self.engine.entered.wait(2))
            start = time.perf_counter()
            self.assertEqual(self.client.get('/v1/health').status_code,200)
            self.assertLess(time.perf_counter()-start,0.5)
            busy = self.client.post('/v1/tts',json={'text':'A second request'})
            self.assertEqual(busy.status_code,503)
            self.assertEqual(busy.headers['Retry-After'],'1')
            self.engine.release.set()
            self.assertEqual(first.result(2).status_code,200)
        self.assertEqual(self.engine.max_active,1)
        self.assertEqual(self.app.state.service.admitted,0)

    def test_custom_style_json_and_multipart(self):
        payload = {'name':'custom', 'style_ttl':{'dims':[1],'data':[1.5],'type':'float32'},
                   'style_dp':{'dims':[1],'data':[1.5],'type':'float32'}}
        r = self.client.post('/v1/styles/import',json=payload)
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(self.client.post('/v1/styles/import',json=payload).status_code,409)
        self.assertEqual(self.client.post('/v1/tts',json={'text':'Hi','voice':'custom'}).status_code,200)
        bad = {**payload,'name':'bad','style_dp':{'dims':[2],'data':[1,2],'type':'float32'}}
        self.assertEqual(self.client.post('/v1/styles/import',json=bad).status_code,400)
        self.assertFalse((Path(self.tmp.name)/'bad.json').exists())
        r = self.client.post('/v1/styles/import',files={'file':('other.json',json.dumps(payload).encode(),'application/json')})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['name'],'other')
        self.assertEqual(self.client.post('/v1/styles/import',json={**payload,'name':'M1'}).status_code,409)

    def test_body_limit(self):
        r = self.client.post('/v1/tts', content=b' '*(2*1024*1024+1), headers={'content-type':'application/json'})
        self.assertEqual(r.status_code,413)
        self.assertEqual(self.engine.calls,[])

    def test_chunking_preserves_content_and_hard_bound(self):
        for text in ['Mr. Smith called. Dr. Lee answered! Why?', '가'*1201,
                     'Short.\n\nAnother paragraph. ' + 'word '*200]:
            chunks = split_text(text, 'ko', 80, 40)
            self.assertTrue(all(0<len(c)<=80 for c in chunks))
            self.assertEqual(''.join(''.join(chunks).split()), ''.join(text.split()))
        self.assertEqual(split_text('Short.\n\nAnother paragraph.','en'),
                         ['Short.','Another paragraph.'])

    def test_synthesis_failure_releases_admission(self):
        self.engine.fail=True
        with self.assertLogs('server', level='ERROR'):
            response = self.client.post('/v1/tts',json={'text':'Hello'})
        self.assertEqual(response.status_code,500)
        self.assertEqual(response.json()['error']['code'],'synthesis_failed')
        self.assertEqual(self.app.state.service.admitted,0)
        self.assertFalse(self.app.state.service.lock.locked())
        self.engine.fail=False
        self.assertEqual(self.client.post('/v1/tts',json={'text':'Hello'}).status_code,200)

    def test_failure_after_stream_start_has_error_without_end(self):
        self.engine.fail_at = 2
        with self.assertLogs('server', level='ERROR'):
            response = self.client.post('/v1/tts',json={
                'text':'First sentence. Second sentence.', 'lang':'en',
                'max_chunk_length':20, 'stream':True, 'stream_format':'sse'})
        self.assertEqual(response.status_code,200)
        self.assertIn('event: audio',response.text)
        self.assertIn('event: error',response.text)
        self.assertNotIn('event: end',response.text)
        self.assertEqual(self.app.state.service.admitted,0)

    def test_failed_header_send_releases_unstarted_stream(self):
        async def check():
            lock=asyncio.Lock();await lock.acquire()
            service=SimpleNamespace(lock=lock,admitted=1,cancellations=set())
            lease=Lease(service)
            async def body():
                yield b'audio'
            response=AudioStreamingResponse(body(),lease=lease)
            async def receive():
                return {'type':'http.disconnect'}
            async def send(message):
                raise OSError('client closed before headers')
            with self.assertRaises(Exception):
                await response({'type':'http','asgi':{'spec_version':'2.4'}},receive,send)
            self.assertTrue(lease.closed)
            self.assertFalse(lock.locked())
            self.assertEqual(service.admitted,0)
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
