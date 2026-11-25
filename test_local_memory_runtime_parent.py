"""Independent parent factory contracts; entirely fictional/toy/offline."""
from pathlib import Path
import sys,socket,tempfile,unittest
from unittest.mock import patch
import bootstrap_test_bot as bot
from memory_core import MemoryStore
from companion_chat import MemoryUnavailable
O='fictional_builder_owner';C='fictional_builder_character';TOKEN='F'*48
class E:
    dimension=2;model_id='fictional-parent-builder'
    def __init__(self):self.calls=0;self.fail=False
    def embed(self,text):
        self.calls+=1
        if self.fail:raise RuntimeError('FICTIONAL_PRIVATE_BUILDER')
        return [1.0,0.2]
class K:
    def __init__(self):self.calls=0
    def decide(self,text):self.calls+=1;return {'key':'喜好','how':'category_only','needs_review':True}
class Parent(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'fictional.sqlite3';self.s=MemoryStore(self.path);self.e=E();self.k=K();self.t=100.;self.runtimes=[]
    def tearDown(self):
        for r in self.runtimes:self.assertTrue(r.worker.stop(timeout=2))
        self.s.close();self.tmp.cleanup()
    def factory(self):
        fn=getattr(bot,'build_local_memory_runtime',None);self.assertTrue(callable(fn),'factory absent in accepted baseline (expected RED)');return fn
    def rt(self,**kw):
        args={'owner_id':O,'character_id':C,'idle_seconds':1,'clock':lambda:self.t};args.update(kw)
        r=self.factory()(self.s,self.e,self.k,TOKEN,**args);self.runtimes.append(r);return r
    def test_construction_is_opt_in_no_inference_network_or_store_close(self):
        with patch.object(socket,'socket',side_effect=AssertionError('no sockets')):
            r=self.rt();self.assertEqual(set(vars(r)),{'backend','indexer','searcher','client','queue','processor','worker'});self.assertFalse(r.worker.running);self.assertEqual((self.k.calls,self.e.calls),(0,0));self.assertEqual(self.s.list_memories(O,C)['total'],0)
            self.assertTrue(r.worker.stop(timeout=2));self.assertTrue(self.s.append_message(O,C,'fictional-after-stop','user','虛構測試後續訊息'))
    def test_real_capture_ack_source_client_and_scope_auth(self):
        with patch.object(socket,'socket',side_effect=AssertionError('no sockets')):
            r=self.rt();other=self.rt(character_id='fictional_other');mid=r.worker.capture('我喜歡虛構星圖攝影');self.t+=2
            self.assertEqual(r.worker.process_once(),{'status':'ok'});self.assertEqual(r.queue.status()['done_count'],1)
            hits=r.client.search(TOKEN,'虛構星圖',3,'real');self.assertEqual(len(hits),1);self.assertEqual(hits[0]['source_ids'],[mid]);self.assertEqual(other.client.search(TOKEN,'虛構星圖',3,'real'),[])
            with patch.object(r.searcher,'search',wraps=r.searcher.search) as spy:
                with self.assertRaises(MemoryUnavailable):r.client.search('X'*48,'虛構星圖',3,'real')
                self.assertEqual(spy.call_count,0)
    def test_native_index_failure_keeps_retry_then_dedup_drains(self):
        r=self.rt();r.worker.capture('我喜歡虛構星圖收藏');self.t+=2;self.e.fail=True
        self.assertEqual(r.worker.process_once(),{'status':'retry','error_category':'index_failed'});self.assertEqual(r.queue.status()['done_count'],0);self.assertEqual(len(self.s.pending_jobs(O,C)),1)
        self.e.fail=False;self.t+=31
        self.assertEqual(r.worker.process_once(),{'status':'ok'});self.assertEqual(self.k.calls,1);self.assertEqual(self.s.list_memories(O,C)['total'],1);self.assertEqual(len(r.client.search(TOKEN,'星圖',3,'real')),1)
    def test_reopen_pending_without_new_chat_recovers(self):
        r=self.rt();mid=r.worker.capture('我喜歡虛構星圖模型');self.assertTrue(r.worker.stop(timeout=2));self.s.close();self.s=MemoryStore(self.path);self.t+=2
        new=self.rt();self.assertEqual(self.k.calls,0);self.assertEqual(new.worker.process_once(),{'status':'ok'});hits=new.client.search(TOKEN,'星圖',3,'real');self.assertEqual(len(hits),1);self.assertEqual(hits[0]['source_ids'],[mid])
    def test_withdrawn_source_not_revived_by_factory_processor(self):
        r=self.rt();text='我喜歡虛構星圖繪畫';r.worker.capture(text);self.t+=2;self.assertEqual(r.worker.process_once(),{'status':'ok'});row=self.s.list_memories(O,C)['items'][0];self.assertTrue(self.s.delete_memory(O,C,row['memory_id']));calls=self.k.calls
        self.assertEqual(r.processor(text),{'status':'ok'});self.assertEqual(self.k.calls,calls);self.assertEqual(r.client.search(TOKEN,'星圖',3,'real'),[])
    def test_bad_configuration_is_sanitized_and_caller_store_survives(self):
        fn=self.factory()
        for bad in ('','short','F'*1025,'F'*32+' ',None):
            with self.subTest(token_length=None if bad is None else len(bad)):
                with self.assertRaises(bot.BootstrapError) as caught:fn(self.s,self.e,self.k,bad,owner_id=O,character_id=C)
                self.assertEqual(str(caught.exception),'local memory runtime unavailable')
        class Bad:
            @property
            def dimension(self):raise RuntimeError('FICTIONAL_PRIVATE_BUILDER')
        with self.assertRaises(bot.BootstrapError) as caught:fn(self.s,Bad(),self.k,TOKEN,owner_id=O,character_id=C)
        self.assertNotIn('FICTIONAL_PRIVATE_BUILDER',str(caught.exception));self.assertTrue(self.s.append_message(O,C,'fictional-after-failure','user','虛構失敗後仍可寫入'))
    def test_bootstrap_error_from_dependency_also_sanitized(self):
        class Bad:
            @property
            def dimension(self):raise bot.BootstrapError('FICTIONAL_PRIVATE_BOOTSTRAP_DETAIL')
        with self.assertRaises(bot.BootstrapError) as caught:self.factory()(self.s,Bad(),self.k,TOKEN,owner_id=O,character_id=C)
        self.assertEqual(str(caught.exception),'local memory runtime unavailable')
        self.assertTrue(self.s.append_message(O,C,'fictional-after-bootstrap-error','user','虛構測試訊息'))
if __name__=='__main__':unittest.main(verbosity=2)
