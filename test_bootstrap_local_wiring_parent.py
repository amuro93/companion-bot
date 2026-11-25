"""Parent integration acceptance: real pipeline; only inference/Telegram are fixtures."""
import contextlib
from pathlib import Path
import socket,sqlite3,tempfile,threading,time,traceback,unittest
from types import SimpleNamespace
from unittest.mock import patch
import bootstrap_test_bot as bot
import memory_core,telegram_test_adapter
from companion_chat import CompanionChat
from memory_deferred_worker import DeferredMemoryWorker

class Embedder:
    dimension=2; model_id='fictional-bootstrap-fixture'
    def embed(self,text):return [1.0,0.2]

class Wiring(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.tmp.name);self.now=100.;self.runtime=None;self.stores=[];self.closes=[];self.key_calls=[];self.model_calls=[];self.factory_calls=0;self.poll_calls=0;self.release=threading.Event();self.entered=threading.Event();self.block=False
        self.stack=contextlib.ExitStack();self.addCleanup(self.cleanup)
        for name,value in [('ROOT',self.root),('DB',self.root/'fictional.sqlite3'),('L0_DIR',self.root),('LocalEmbedder',Embedder),('load_persona',lambda:'隔離虛構人物'),('model_reply',self.model)]:self.stack.enter_context(patch.object(bot,name,value))
        self.original_factory=bot.build_local_memory_runtime
        self.stack.enter_context(patch.object(bot,'build_local_memory_runtime',self.factory))
        original_store=memory_core.MemoryStore
        def make_store(*a,**kw):
            s=original_store(*a,**kw);self.stores.append(s);old=s.close
            def close():self.closes.append(s);return old()
            s.close=close;return s
        self.stack.enter_context(patch.object(memory_core,'MemoryStore',make_store))
        self.stack.enter_context(patch.object(telegram_test_adapter,'build_test_application',self.application))
        self.net=self.stack.enter_context(patch.object(socket,'socket',side_effect=AssertionError('network forbidden')))
        self.poll=lambda chat,admin:None
    def cleanup(self):
        self.release.set()
        if self.runtime:self.assertTrue(self.runtime.worker.stop(timeout=3))
        self.stack.close()
        for s in self.stores:
            if s not in self.closes:s.close()
        self.tmp.cleanup()
    def model(self,messages):
        self.model_calls.append(messages)
        if messages and '記憶守門員' in messages[0]['content']:
            self.key_calls.append(threading.current_thread().name)
            if self.block:self.entered.set();self.release.wait(3)
            return '喜好/茶'
        return '虛構回覆'
    def factory(self,*a,**kw):
        self.factory_calls+=1;kw.update(idle_seconds=1,clock=lambda:self.now)
        self.runtime=self.original_factory(*a,**kw);self.runtime.worker.poll_seconds=.01
        return self.runtime
    def application(self,token,chat,admin):
        self.assertIsInstance(chat,CompanionChat);self.chat=chat
        def polling(**kw):
            self.poll_calls+=1;self.assertEqual(kw,{'drop_pending_updates':True});self.assertIsNotNone(self.runtime,'local runtime must be assembled before polling');self.assertTrue(self.runtime.worker.running);self.assertIs(chat.memory_client,self.runtime.client);self.poll(chat,admin)
        return SimpleNamespace(run_polling=polling)
    def run_bot(self):return bot.run({'BOT_TELEGRAM_TOKEN':'fictional-token','BOT_TELEGRAM_USER_ID':'42'})
    def wait_for(self,fn):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if fn():return
            time.sleep(.01)
        self.fail('bounded wait timed out')
    def update(self,text):return SimpleNamespace(effective_user=SimpleNamespace(id=42),effective_chat=SimpleNamespace(type='private',id=42),message=SimpleNamespace(text=text))
    def assert_clean_error(self,fn):
        try:fn()
        except bot.BootstrapError as exc:
            self.assertNotIn('FICTIONAL_SECRET', ''.join(traceback.format_exception(exc)))
        else:self.fail('expected content-free BootstrapError')
    def test_actual_chat_captures_then_background_indexes_and_searches(self):
        def poll(chat,admin):
            self.assertEqual(chat.turn(self.update('我喜歡虛構茉莉花茶')),'虛構回覆');self.assertEqual(self.key_calls,[])
            self.assertEqual(self.runtime.queue.status()['done_count'],0)
            self.assertEqual(self.stores[0].list_memories(bot.OWNER,bot.CHARACTER)['total'],0)
            self.now+=2
            self.wait_for(lambda:self.runtime.queue.status()['done_count']==1)
            self.assertEqual(self.key_calls,['DeferredMemoryWorker'])
            hits=self.runtime.client.search(next(iter(chat.scopes.values())).token,'虛構茶',3,'real');self.assertEqual(len(hits),1)
            with contextlib.closing(sqlite3.connect(bot.DB)) as c:
                rows=c.execute('SELECT role,content FROM messages').fetchall()
            self.assertIn(('assistant','虛構回覆'),rows);self.assertIn(('user','我喜歡虛構茉莉花茶'),rows)
            self.assertEqual(len(hits[0]['source_ids']),1)
        self.poll=poll;self.run_bot();self.assertEqual(self.factory_calls,1);self.assertEqual(self.poll_calls,1);self.assertFalse(self.runtime.worker.running);self.assertEqual(len(self.closes),1);self.assertEqual(self.net.call_count,0)
    def test_polling_error_stops_before_close_and_sanitizes(self):
        def fail(*a):raise RuntimeError('FICTIONAL_SECRET_poll')
        self.poll=fail;self.assert_clean_error(self.run_bot);self.assertIsNotNone(self.runtime);self.assertFalse(self.runtime.worker.running);self.assertEqual(len(self.closes),1)
    def test_startup_model_probe_failure_closes_without_poll(self):
        with patch.object(bot,'model_reply',side_effect=RuntimeError('FICTIONAL_SECRET_model')):self.assert_clean_error(self.run_bot)
        self.assertEqual(self.poll_calls,0);self.assertEqual(len(self.closes),1);self.assertFalse(self.runtime.worker.running)
    def test_runtime_build_failure_closes_owned_store(self):
        with patch.object(bot,'build_local_memory_runtime',side_effect=RuntimeError('FICTIONAL_SECRET_factory')):self.assert_clean_error(self.run_bot)
        self.assertEqual(self.poll_calls,0);self.assertEqual(len(self.closes),1)
    def test_busy_real_processor_does_not_close_store_or_claim_cancel(self):
        self.block=True
        def poll(chat,admin):
            chat.turn(self.update('我喜歡虛構茉莉花茶'));self.now+=2;self.assertTrue(self.entered.wait(2))
            # A second real chat turn remains responsive while extraction is blocked.
            self.assertEqual(chat.turn(self.update('我也喜歡虛構藍茶')),'虛構回覆')
            self.assertEqual(len(self.key_calls),1);self.assertFalse(self.release.is_set())
        self.poll=poll;original_stop=DeferredMemoryWorker.stop
        def short_stop(worker,timeout=5):return original_stop(worker,timeout=.01)
        with patch.object(DeferredMemoryWorker,'stop',short_stop):self.assert_clean_error(self.run_bot)
        self.assertTrue(self.runtime.worker.running);self.assertEqual(self.closes,[])
        self.assertTrue(self.stores[0].append_message(bot.OWNER,bot.CHARACTER,'fictional-open','assistant','still open'))
        self.release.set();self.assertTrue(original_stop(self.runtime.worker,timeout=3));self.assertEqual(self.runtime.queue.status()['done_count'],1)
    def test_stop_exception_is_sanitized_store_retained(self):
        original_stop=DeferredMemoryWorker.stop
        def stop(worker,timeout=5):
            original_stop(worker,timeout=timeout);raise RuntimeError('FICTIONAL_SECRET_stop')
        with patch.object(DeferredMemoryWorker,'stop',stop):self.assert_clean_error(self.run_bot)
        self.assertEqual(self.closes,[]);self.assertTrue(self.stores[0].append_message(bot.OWNER,bot.CHARACTER,'fictional-open','assistant','still open'))
    def test_thread_start_failure_is_sanitized_and_closes(self):
        with patch.object(threading.Thread,'start',side_effect=RuntimeError('FICTIONAL_SECRET_thread')):self.assert_clean_error(self.run_bot)
        self.assertEqual(self.poll_calls,0);self.assertEqual(len(self.closes),1);self.assertIsNotNone(self.runtime);self.assertFalse(self.runtime.worker.running)
    def test_store_close_exception_sanitized(self):
        def poll(chat,admin):
            old=self.stores[0].close
            def close():old();raise RuntimeError('FICTIONAL_SECRET_close')
            self.stores[0].close=close
        self.poll=poll;self.assert_clean_error(self.run_bot)
    def test_keyboard_interrupt_stops_and_closes(self):
        def poll(*a):raise KeyboardInterrupt()
        self.poll=poll
        with self.assertRaises(KeyboardInterrupt):self.run_bot()
        self.assertFalse(self.runtime.worker.running);self.assertEqual(len(self.closes),1)
    def test_pending_restart_processed_without_new_chat(self):
        # First actual run persists but does not advance the idle deadline.
        self.poll=lambda chat,admin:chat.turn(self.update('我喜歡虛構茉莉花茶'))
        self.run_bot();self.assertEqual(self.runtime.queue.status()['done_count'],0);self.assertEqual(self.key_calls,[])
        self.now+=2
        def poll(chat,admin):
            self.wait_for(lambda:self.runtime.queue.status()['done_count']==1)
            self.assertEqual(len(self.runtime.client.search(next(iter(chat.scopes.values())).token,'虛構茶',3,'real')),1)
        self.poll=poll;self.run_bot();self.assertEqual(len(self.key_calls),1);self.assertEqual(self.factory_calls,2);self.assertEqual(len(self.closes),2)
    def test_preflight_ignores_retired_urls_but_not_model_url(self):
        with patch.object(bot,'QDRANT_URL','retired-invalid'),patch.object(bot,'DECISION_URL','retired-invalid'):
            self.assertEqual(bot.preflight({'BOT_TELEGRAM_TOKEN':'fictional','BOT_TELEGRAM_USER_ID':'42'}),('fictional',42))
        with patch.object(bot,'MODEL_URL','http://example.invalid:18080'):
            with self.assertRaises(bot.BootstrapError):bot.preflight({'BOT_TELEGRAM_TOKEN':'fictional','BOT_TELEGRAM_USER_ID':'42'})

if __name__=='__main__':unittest.main(verbosity=2)
