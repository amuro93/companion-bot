"""Parent permanent regression: configuration isolation with real memory machinery.
Only external inference/embedding/Telegram transport use fixtures. Synthetic data only.
"""
from pathlib import Path
import sys,unittest,json,tempfile,contextlib,socket,time,sqlite3,dataclasses,threading
from types import SimpleNamespace
from unittest.mock import patch
import bootstrap_test_bot as bot
import os
SCRATCH=Path(os.environ.get('TMPDIR',tempfile.gettempdir()))
class Embedder:
 dimension=2;model_id='independent-fictional-embedder'
 def embed(self,text):return [1.,.2]
class Response:
 status=200
 def __init__(self,content):self.body=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}]}).encode()
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def read(self,n):return self.body
class ParentConfigAcceptance(unittest.TestCase):
 def api(self):
  self.assertTrue(callable(getattr(bot,'default_config',None)),'A5e lacks per-bot configuration contract')
  import bot_config
  return bot_config
 def fixture(self,base,label='alpha',family=None):
  root=base/label;root.mkdir();(root/'l0').mkdir();(root/'persona.txt').write_text('虛構人格-'+label,encoding='utf8');(root/'l0/fact.md').write_text('> load: always\n# 虛構L0\n獨有虛構L0-'+label,encoding='utf8')
  return {'root':label,'db':'memory.sqlite3','owner_id':'fictional-owner-'+label,'character_id':'fictional-char-'+label,'persona_file':'persona.txt','l0_dir':'l0','model_url':'http://127.0.0.1:'+('19101' if label=='alpha' else '19102')+'/v1/chat/completions','model_id':'fictional-model-'+label,'telegram_token_env':label.upper()+'_TOKEN','telegram_user_id_env':label.upper()+'_USER','addressee_family_key':family}
 def save(self,base,data,name='config.json'):
  p=base/name;p.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8');return p
 def test_loader_paths_and_no_side_effects(self):
  cfgmod=self.api()
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d);raw=self.fixture(base);p=self.save(base,raw)
   reads=[];original=Path.read_text;original_bytes=Path.read_bytes
   def read(path,*a,**kw):reads.append(path.resolve());return original(path,*a,**kw)
   def read_bytes(path,*a,**kw):reads.append(path.resolve());return original_bytes(path,*a,**kw)
   with patch.object(Path,'read_text',read),patch.object(Path,'read_bytes',read_bytes),patch.object(socket,'socket',side_effect=AssertionError('network forbidden')) as net,patch.object(sqlite3,'connect',side_effect=AssertionError('DB forbidden')) as db,patch.object(threading.Thread,'start',side_effect=AssertionError('thread forbidden')) as start:
    cfg=cfgmod.load_bot_config(p)
   self.assertEqual(reads,[p.resolve()]);self.assertEqual((net.call_count,db.call_count,start.call_count),(0,0,0));self.assertFalse((base/'alpha/memory.sqlite3').exists())
   self.assertEqual(cfg.root,base/'alpha');self.assertEqual(cfg.db,base/'alpha/memory.sqlite3');self.assertEqual(cfg.persona_file,base/'alpha/persona.txt');self.assertEqual(cfg.l0_dir,base/'alpha/l0');self.assertIsNone(cfg.addressee_family_key)
   with self.assertRaises(dataclasses.FrozenInstanceError):cfg.owner_id='other'
 def test_invalid_schema_paths_urls_and_direct_constructor(self):
  cfgmod=self.api()
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d);raw=self.fixture(base)
   invalid=[{**raw,'extra':'no'},{k:v for k,v in raw.items() if k!='owner_id'},{**raw,'telegram_token_env':'BAD NAME'},{**raw,'telegram_user_id_env':'ALPHA_TOKEN'},{**raw,'owner_id':'bad\nid'},{**raw,'character_id':None},{**raw,'db':'../outside.sqlite3'},{**raw,'db':'C:relative.db'},{**raw,'db':'//server/share.db'},{**raw,'persona_file':'http://example.invalid/x'},{**raw,'l0_dir':''},{**raw,'model_id':'bad\nmodel'},{**raw,'addressee_family_key':'家人/子女'}]
   invalid += [{**raw,'model_url':u} for u in ['https://127.0.0.1:19101/v1/chat/completions','http://example.invalid:19101/v1/chat/completions','http://localhost:19101/v1/chat/completions','http://127.0.0.1:19101/v1/chat/completions?q=x','http://127.0.0.1:19101/v1/chat/completions#x','http://username:password@127.0.0.1:19101/v1/chat/completions','http://127.0.0.1:19101/other','http://127.0.0.1:65536/v1/chat/completions']]
   for item in invalid:
    with self.subTest(item=list(item)):
     with self.assertRaises(cfgmod.ConfigError):cfgmod.load_bot_config(self.save(base,item))
   p=base/'duplicate.json';p.write_text(json.dumps(raw)[:-1]+',"owner_id":"duplicate"}',encoding='utf8')
   with self.assertRaises(cfgmod.ConfigError):cfgmod.load_bot_config(p)
   valid=cfgmod.load_bot_config(self.save(base,raw))
   with self.assertRaises(cfgmod.ConfigError):dataclasses.replace(valid,model_url='http://evil.invalid:1/v1/chat/completions')
   with self.assertRaises(cfgmod.ConfigError):dataclasses.replace(valid,db=base/'outside.sqlite3')
 def test_strict_validation_boundary(self):
  cfgmod=self.api()
  import traceback
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d);raw=self.fixture(base);valid=cfgmod.load_bot_config(self.save(base,raw))
   cases=[('telegram_token_env','TOKEN\n'),('telegram_user_id_env','alpha_token'),('db','memory.sqlite3:stream'),('db','C:'),('db',' memory.sqlite3'),('db','memory.sqlite3 '),('db','CON'),('db','dir/../memory.sqlite3'),('db','file.'),('db','bad?.sqlite3'),('db','bad*.sqlite3'),('db','bad|name.sqlite3'),('db','bad<name.sqlite3'),('db','bad>name.sqlite3'),('db','bad"name.sqlite3'),('model_id','bad\x7f'),('model_url','http://127.0.0.1:FICTIONAL_SECRET/v1/chat/completions'),('model_url','http://[FICTIONAL_SECRET/v1/chat/completions'),('addressee_family_key','家人/子女')]
   for field,value in cases:
    with self.subTest(field=field,value=value):
     with self.assertRaises(cfgmod.ConfigError):cfgmod.load_bot_config(self.save(base,{**raw,field:value}))
     with self.assertRaises(cfgmod.ConfigError):dataclasses.replace(valid,**{field:value})
   broken=base/'utf8.json';broken.write_bytes(b'{"x":"FICTIONAL_SECRET\xff"}')
   try:cfgmod.load_bot_config(broken)
   except cfgmod.ConfigError as exc:self.assertNotIn('FICTIONAL_SECRET',''.join(traceback.format_exception(exc)))
   else:self.fail('invalid UTF8 config accepted')
 def test_windows_junction_escape_and_path_alias_rejection(self):
  cfgmod=self.api()
  if sys.platform != 'win32':self.skipTest('Windows junction-only test')
  import _winapi
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d);raw=self.fixture(base);outside=base/'outside';outside.mkdir();link=base/'alpha/link'
   _winapi.CreateJunction(str(outside),str(link))
   try:
    for field,value in [('db','link/escaped.sqlite3'),('persona_file','link/persona.txt'),('l0_dir','link')]:
     with self.subTest(field=field):
      with self.assertRaises(cfgmod.ConfigError):cfgmod.load_bot_config(self.save(base,{**raw,field:value}))
   finally:link.rmdir()
 def test_model_payload_and_env_isolation(self):
  cfgmod=self.api()
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d)
   for label in ['alpha','beta']:
    raw=self.fixture(base,label);cfg=cfgmod.load_bot_config(self.save(base,raw))
    calls=[]
    def opener(req,timeout):calls.append((req,timeout));return Response('fixture response')
    self.assertEqual(bot.model_reply([{'role':'user','content':'fictional prompt'}],config=cfg,opener=opener),'fixture response')
    self.assertEqual(len(calls),1);req,timeout=calls[0];self.assertEqual(req.full_url,raw['model_url']);self.assertEqual(timeout,40)
    payload=json.loads(req.data);self.assertEqual(payload,{'model':raw['model_id'],'messages':[{'role':'user','content':'fictional prompt'}],'chat_template_kwargs':{'enable_thinking':False},'max_tokens':512,'stream':False});self.assertNotIn('Authorization',req.headers)
    self.assertEqual(bot.preflight({raw['telegram_token_env']:'fictional-token',raw['telegram_user_id_env']:'52'},config=cfg),('fictional-token',52))
    with self.assertRaises(bot.BootstrapError):bot.preflight({'BOT_TELEGRAM_TOKEN':'wrong','BOT_TELEGRAM_USER_ID':'42'},config=cfg)
 def test_default_transport_refuses_redirect_without_network(self):
  cfgmod=self.api()
  import urllib.request,urllib.response,io,email.message
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d);raw=self.fixture(base);cfg=cfgmod.load_bot_config(self.save(base,raw));seen=[]
   def transport(handler,request):
    seen.append(request.full_url);headers=email.message.Message();headers['Location']='http://external.invalid/leak'
    response=urllib.response.addinfourl(io.BytesIO(b''),headers,request.full_url,302);response.msg='Found';return response
   with patch.object(urllib.request.HTTPHandler,'http_open',transport),patch.object(socket,'socket',side_effect=AssertionError('network forbidden')) as net:
    with self.assertRaises(bot.BootstrapError):bot.model_reply([{'role':'user','content':'fictional private prompt'}],config=cfg)
   self.assertEqual(seen,[cfg.model_url]);self.assertEqual(net.call_count,0)
 def test_main_selected_config_real_pipeline_two_roles(self):
  cfgmod=self.api()
  import memory_core,telegram_test_adapter
  from companion_chat import CompanionChat
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   base=Path(d);snapshots=[]
   for label,family in [('alpha',None),('beta','家人/配偶')]:
    raw=self.fixture(base,label,family);path=self.save(base,raw,label+'.json');cfg=cfgmod.load_bot_config(path);now=[100.];runtimes=[];requests=[];stores=[];closed=[]
    real_factory=bot.build_local_memory_runtime;real_store=memory_core.MemoryStore
    def factory(*args,**kw):
     self.assertEqual(kw.get('owner_id'),cfg.owner_id);self.assertEqual(kw.get('character_id'),cfg.character_id);self.assertEqual(kw.get('idle_seconds',180),180)
     kw.update(idle_seconds=1,clock=lambda:now[0]);runtime=real_factory(*args,**kw);runtime.worker.poll_seconds=.01;runtimes.append(runtime);return runtime
    def store_factory(db):
     self.assertEqual(Path(db),cfg.db);s=real_store(db);stores.append(s);original=s.close
     def close():closed.append(True);return original()
     s.close=close;return s
    def model_opener(req,timeout):
     self.assertEqual(req.full_url,cfg.model_url);payload=json.loads(req.data);self.assertEqual(payload['model'],cfg.model_id);requests.append(payload)
     return Response('喜好/茶' if payload['messages'] and '記憶守門員' in payload['messages'][0]['content'] else '虛構回應-'+label)
    def app(token,chat,admin):
     self.assertEqual(token,'fixture-'+label);self.assertIsInstance(chat,CompanionChat);self.assertEqual(chat.persona,'虛構人格-'+label);self.assertEqual((admin.owner_id,admin.character_id),(cfg.owner_id,cfg.character_id));self.assertEqual(chat.persona_l0.base_dir,cfg.l0_dir)
     scope=chat.scopes[52];self.assertEqual((scope.owner_id,scope.character_id),(cfg.owner_id,cfg.character_id))
     def polling(**kw):
      self.assertEqual(kw,{'drop_pending_updates':True});runtime=runtimes[0];self.assertTrue(runtime.worker.running)
      self.assertEqual(runtime.processor._keyer.addressee_family_key,family)
      update=SimpleNamespace(effective_user=SimpleNamespace(id=52),effective_chat=SimpleNamespace(type='private',id=52),message=SimpleNamespace(text='我喜歡虛構茉莉花茶'))
      self.assertEqual(chat.turn(update),'虛構回應-'+label);self.assertEqual(chat.record_failures,0);now[0]+=2
      end=time.monotonic()+3
      while runtime.queue.status()['done_count']!=1 and time.monotonic()<end:time.sleep(.01)
      self.assertEqual(runtime.queue.status()['done_count'],1);self.assertEqual(admin.listing()['total'],1)
      hits=runtime.client.search(scope.token,'虛構茶',3,'real');self.assertEqual(len(hits),1)
      self.assertEqual(stores[0].list_memories('wrong-owner',cfg.character_id)['total'],0);self.assertEqual(stores[0].list_memories(cfg.owner_id,'wrong-character')['total'],0)
     return SimpleNamespace(run_polling=polling)
    original_model=bot.model_reply
    def inference(messages,*,opener=None,**kw):return original_model(messages,opener=model_opener,**kw)
    # Only embedding/inference/Telegram boundaries replaced; all memory machinery real.
    with patch.object(bot,'LocalEmbedder',Embedder),patch.object(bot,'build_local_memory_runtime',factory),patch.object(memory_core,'MemoryStore',store_factory),patch.object(telegram_test_adapter,'build_test_application',app),patch.object(bot,'model_reply',inference),patch.dict('os.environ',{raw['telegram_token_env']:'fixture-'+label,raw['telegram_user_id_env']:'52'},clear=True),patch.object(socket,'socket',side_effect=AssertionError('network forbidden')) as net:
     try:bot.main(['--config',str(path)])
     finally:
      for r in runtimes:r.worker.stop(timeout=3)
    self.assertEqual(net.call_count,0);self.assertEqual(closed,[True]);self.assertFalse(runtimes[0].worker.running)
    with contextlib.closing(sqlite3.connect(cfg.db)) as conn:
     rows=conn.execute('SELECT owner_id,character_id,role,content FROM messages').fetchall()
    self.assertEqual(len(rows),2);self.assertEqual({(r[0],r[1]) for r in rows},{(cfg.owner_id,cfg.character_id)});self.assertEqual({r[2] for r in rows},{'user','assistant'});self.assertGreaterEqual(len(requests),3);chat_payloads=[p for p in requests if any('虛構人格-'+label in m['content'] for m in p['messages'])];self.assertEqual(len(chat_payloads),1);self.assertTrue(any('獨有虛構L0-'+label in m['content'] for m in chat_payloads[0]['messages']));snapshots.append((cfg.db,rows))
   self.assertNotEqual(snapshots[0][0],snapshots[1][0]);self.assertNotEqual(snapshots[0][1],snapshots[1][1])
if __name__=='__main__':unittest.main(verbosity=2)
