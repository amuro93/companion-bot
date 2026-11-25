"""Independent parent verification. Synthetic data; real HTTP and SQLite."""
import unittest, tempfile, os, json, threading, contextlib, time, traceback, sqlite3, dataclasses, socket
from pathlib import Path
from types import SimpleNamespace
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch
import bootstrap_test_bot as bot
import bot_config as bc
SCRATCH=Path(os.environ.get('TMPDIR') or tempfile.gettempdir())

class ParentAuth(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=SCRATCH);self.root=Path(self.tmp.name)
  (self.root/'persona.txt').write_text('虛構測試角色，無真人資訊。',encoding='utf8');(self.root/'l0').mkdir()
  self.hits=[];self.expected='fictional-key-alpha';self.redirect=None
  parent=self
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    parent.hits.append((self.path,self.headers.get('Authorization'),body))
    if parent.redirect:
     self.send_response(parent.redirect);self.send_header('Location',parent.sink_url);self.send_header('Content-Length','0');self.end_headers();return
    if self.headers.get('Authorization')!='Bearer '+parent.expected:
     self.send_response(401);self.send_header('Content-Length','0');self.end_headers();return
    content='喜好/茶' if body['messages'] and '記憶守門員' in body['messages'][0]['content'] else '虛構回應'
    data=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}]}).encode()
    self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
   def do_GET(self):parent.sink_hits.append(True);self.send_response(200);self.end_headers()
  self.sink_hits=[]
  self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.sink=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  self.threads=[]
  for s in [self.server,self.sink]:
   t=threading.Thread(target=s.serve_forever,kwargs={'poll_interval':.01},daemon=True);t.start();self.threads.append(t)
  self.url=f'http://127.0.0.1:{self.server.server_port}/v1/chat/completions';self.sink_url=f'http://127.0.0.1:{self.sink.server_port}/sink'
 def tearDown(self):
  for s in [self.server,self.sink]:s.shutdown();s.server_close()
  for t in self.threads:t.join(2)
  self.tmp.cleanup()
 def cfg(self,**kw):
  self.assertIn('model_api_key_env',bc.BotConfig.__dataclass_fields__,'baseline lacks auth contract')
  raw=dict(root=self.root,db='fictional.sqlite3',owner_id='fictional-owner',character_id='fictional-character',persona_file='persona.txt',l0_dir='l0',model_url=self.url,model_id='fictional-model',telegram_token_env='FICTIONAL_TG_TOKEN',telegram_user_id_env='FICTIONAL_TG_USER',model_api_key_env='FICTIONAL_MODEL_KEY')
  raw.update(kw);return bc.BotConfig(**raw)
 def reply(self,cfg,env):return bot.model_reply([{'role':'user','content':'虛構測試短句'}],config=cfg,env=env)
 def test_01_actual_header_and_wrong_key(self):
  cfg=self.cfg();self.assertEqual(self.reply(cfg,{'FICTIONAL_MODEL_KEY':self.expected}),'虛構回應')
  with self.assertRaises(bot.BootstrapError):self.reply(cfg,{'FICTIONAL_MODEL_KEY':'wrong-fictional'})
  self.assertEqual(len(self.hits),2);self.assertEqual(self.hits[0][1],'Bearer '+self.expected)
  body=self.hits[0][2];self.assertEqual(body['max_tokens'],512);self.assertIs(body['stream'],False);self.assertEqual(body['chat_template_kwargs'],{'enable_thinking':False})
 def test_02_missing_invalid_no_network_and_redaction(self):
  cfg=self.cfg();marker='FICTIONAL_PRIVATE_MARKER'
  for val in [None,'',42,marker+'\r\nx',marker+' ','中'+marker,marker+'\x00',marker*300]:
   env={} if val is None else {'FICTIONAL_MODEL_KEY':val}
   with patch.dict(os.environ,{'FICTIONAL_MODEL_KEY':self.expected}):
    try:self.reply(cfg,env)
    except bot.BootstrapError as exc:
     self.assertEqual(str(exc),'model unavailable');self.assertNotIn(marker,''.join(traceback.format_exception(exc)))
    else:self.fail('invalid credential accepted')
  self.assertEqual(self.hits,[])
 def test_03_all_redirects_never_forward_auth(self):
  cfg=self.cfg()
  for code in [301,302,303,307,308]:
   self.redirect=code
   with self.assertRaises(bot.BootstrapError):self.reply(cfg,{'FICTIONAL_MODEL_KEY':self.expected})
  self.assertEqual(len(self.hits),5);self.assertEqual(self.sink_hits,[])
 def test_04_exact_remote_allowlist(self):
  remote='http://llm-host:8080/v1/chat/completions'
  self.assertEqual(self.cfg(model_url=remote).model_url,remote)
  for url in [remote.replace(':8080',':8081'),remote.replace('llm-host','evil.invalid'),remote.replace('llm-host','llm-host.evil.invalid'),remote.replace('llm-host','100.64.0.1'),remote.replace('llm-host','LLM-HOST'),remote.replace('llm-host','llm-host.'),remote+'?',remote+'#',remote+'?x',remote+'#x',remote.replace('http:','https:'),remote.replace(':8080',':08080'),remote.replace('llm-host','user:pass@llm-host'),remote.replace('/v1/','/%76%31/')]:
   with self.subTest(url=url):
    with self.assertRaises(bc.ConfigError):self.cfg(model_url=url)
  with self.assertRaises(bc.ConfigError):self.cfg(model_url=remote,model_api_key_env=None)
 def test_05_loader_does_not_access_env_or_network(self):
  cfg=self.cfg();raw={f.name:getattr(cfg,f.name) for f in dataclasses.fields(cfg)}
  raw={k:str(v) if isinstance(v,Path) else v for k,v in raw.items()};p=self.root/'config.json';p.write_text(json.dumps(raw),encoding='utf8')
  class DeniedEnv(dict):
   def get(self,*a):raise AssertionError('env read by loader')
   def __getitem__(self,*a):raise AssertionError('env read by loader')
  with patch.object(os,'environ',DeniedEnv()),patch.object(socket,'socket',side_effect=AssertionError('network')) as net,patch.object(sqlite3,'connect',side_effect=AssertionError('db')) as db:
   self.assertEqual(bc.load_bot_config(p).model_api_key_env,'FICTIONAL_MODEL_KEY')
  self.assertEqual((net.call_count,db.call_count),(0,0))
 def test_06_revalidate_destination_before_key(self):
  cfg=self.cfg();object.__setattr__(cfg,'model_url',self.sink_url)
  with self.assertRaises(bot.BootstrapError):self.reply(cfg,{'FICTIONAL_MODEL_KEY':self.expected})
  self.assertEqual(self.hits,[]);self.assertEqual(self.sink_hits,[])
 def test_07_env_names_and_preflight(self):
  cfg=self.cfg()
  for name in ['',42,'BAD NAME','BAD\n','FICTIONAL_TG_TOKEN','fictional_tg_user']:
   with self.assertRaises(bc.ConfigError):self.cfg(model_api_key_env=name)
  env={'FICTIONAL_TG_TOKEN':'fictional-token','FICTIONAL_TG_USER':'42'}
  with patch.object(bot,'LocalEmbedder',side_effect=AssertionError('heavy')) as heavy:
   with self.assertRaises(bot.BootstrapError):bot.run(env,config=cfg)
  self.assertEqual(heavy.call_count,0);self.assertEqual(self.hits,[])
  self.assertEqual(bot.preflight({**env,'FICTIONAL_MODEL_KEY':self.expected},config=cfg),('fictional-token',42))
 def test_08_keys_not_cached_or_cross_bot(self):
  cfg=self.cfg();env={'FICTIONAL_MODEL_KEY':self.expected};self.reply(cfg,env)
  other=self.cfg(model_api_key_env='OTHER_MODEL_KEY');self.expected='fictional-key-beta';self.reply(other,{'OTHER_MODEL_KEY':self.expected,'FICTIONAL_MODEL_KEY':'wrong'})
  env['FICTIONAL_MODEL_KEY']=self.expected;self.reply(cfg,env)
  self.assertEqual([h[1] for h in self.hits],['Bearer fictional-key-alpha','Bearer fictional-key-beta','Bearer fictional-key-beta'])
 def test_09_noauth_never_reads_secret_and_no_proxy(self):
  cfg=self.cfg(model_api_key_env=None);captured=[]
  class Denied(dict):
   def get(self,*a):raise AssertionError('noauth env lookup')
  def opener(req,timeout):
   captured.append(req.get_header('Authorization'));self.assertEqual(timeout,40)
   class Response:
    status=200
    def __enter__(s):return s
    def __exit__(s,*a):pass
    def read(s,n):return b'{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}'
   return Response()
  self.assertEqual(bot.model_reply([],config=cfg,env=Denied(),opener=opener),'ok');self.assertEqual(captured,[None])
  handlers=[h for h in bot._NO_REDIRECT_OPENER.handlers if isinstance(h,bot.urllib.request.ProxyHandler)]
  self.assertFalse(any(h.proxies for h in handlers),'ambient proxy must not receive keys')
 def test_10_real_run_readiness_chooser_chat_isolated_env(self):
  cfg=self.cfg();import memory_core,telegram_test_adapter
  runtimes=[];stores=[];now=[100.];factory=bot.build_local_memory_runtime;real_store=memory_core.MemoryStore
  class Embedder:
   dimension=2;model_id='fictional-embedder'
   def embed(self,text):return [1.,.2]
  def build(*a,**kw):
   self.assertEqual(kw.get('idle_seconds',180),180);kw.update(idle_seconds=1,clock=lambda:now[0]);r=factory(*a,**kw);r.worker.poll_seconds=.01;runtimes.append(r);return r
  def store(path):s=real_store(path);stores.append(s);return s
  def app(token,chat,admin):
   def poll(**kw):
    update=SimpleNamespace(effective_user=SimpleNamespace(id=42),effective_chat=SimpleNamespace(type='private',id=42),message=SimpleNamespace(text='我喜歡虛構茉莉花茶'))
    self.assertEqual(chat.turn(update),'虛構回應');now[0]+=2
    end=time.monotonic()+4
    while runtimes[0].queue.status()['done_count']!=1 and time.monotonic()<end:time.sleep(.01)
    self.assertEqual(runtimes[0].queue.status()['done_count'],1);self.assertEqual(admin.listing()['total'],1)
   return SimpleNamespace(run_polling=poll)
  env={'FICTIONAL_MODEL_KEY':self.expected,'FICTIONAL_TG_TOKEN':'fictional-token','FICTIONAL_TG_USER':'42'}
  with patch.object(bot,'LocalEmbedder',Embedder),patch.object(bot,'build_local_memory_runtime',build),patch.object(memory_core,'MemoryStore',store),patch.object(telegram_test_adapter,'build_test_application',app),patch.dict(os.environ,{'FICTIONAL_MODEL_KEY':'wrong-global'},clear=False):
   try:bot.run(env,config=cfg)
   finally:
    for r in runtimes:r.worker.stop(timeout=3)
    for s in stores:s.close()
  self.assertGreaterEqual(len(self.hits),3);self.assertTrue(all(h[1]=='Bearer '+self.expected for h in self.hits));self.assertFalse(runtimes[0].worker.running)
  with contextlib.closing(sqlite3.connect(cfg.db)) as conn:rows=conn.execute('SELECT owner_id,character_id,role FROM messages').fetchall()
  self.assertEqual(len(rows),2);self.assertEqual({r[:2] for r in rows},{(cfg.owner_id,cfg.character_id)});self.assertEqual({r[2] for r in rows},{'user','assistant'})

 def test_11_auth_validation_exceptions_sanitized(self):
  cfg=self.cfg();marker='FICTIONAL_'+'PRIVATE_AUTH_EXCEPTION'
  errors=[RuntimeError,bot.BootstrapError]
  class ThrowingEnv(dict):
   def get(self,name,default=None):
    if name=='FICTIONAL_MODEL_KEY':raise error_type(marker)
    return super().get(name,default)
  env=ThrowingEnv(FICTIONAL_TG_TOKEN='fictional-token',FICTIONAL_TG_USER='42')
  for error_type in errors:
   for index,call in enumerate([lambda:bot.preflight(env,config=cfg),lambda:self.reply(cfg,env)]):
    with self.subTest(error_type=error_type.__name__,path=index):
     try:call()
     except Exception as exc:
      self.assertIsInstance(exc,bot.BootstrapError);self.assertNotIn(marker,''.join(traceback.format_exception(exc)))
     else:self.fail('throwing credential source accepted')
  class ThrowingString(str):
   def isascii(self):raise RuntimeError(marker)
  try:self.reply(cfg,{'FICTIONAL_MODEL_KEY':ThrowingString('fictional')})
  except Exception as exc:
   self.assertIsInstance(exc,bot.BootstrapError);self.assertNotIn(marker,''.join(traceback.format_exception(exc)))
  else:self.fail('throwing invalid key accepted')
  self.assertEqual(self.hits,[])

 def test_12_ambient_proxy_cannot_receive_actual_request(self):
  cfg=self.cfg();import importlib
  try:
   with patch.dict(os.environ,{'HTTP_PROXY':self.sink_url,'http_proxy':self.sink_url,'HTTPS_PROXY':self.sink_url,'https_proxy':self.sink_url,'ALL_PROXY':self.sink_url,'all_proxy':self.sink_url,'NO_PROXY':'','no_proxy':''}):
    importlib.reload(bot)
    self.assertEqual(self.reply(cfg,{'FICTIONAL_MODEL_KEY':self.expected}),'虛構回應')
  finally:importlib.reload(bot)
  self.assertEqual(len(self.hits),1);self.assertEqual(self.hits[0][0],'/v1/chat/completions');self.assertEqual(self.sink_hits,[])

if __name__=='__main__':unittest.main(verbosity=2)
