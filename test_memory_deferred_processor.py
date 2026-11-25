"""Parent-owned independent contracts; fictional data/toy inference only."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from memory_core import MemoryStore
from memory_auto import record_user_message_keyed
from memory_index import IndexWorker
from memory_sqlite_vector import SQLiteVectorBackend,SQLiteSemanticSearcher
from memory_deferred_queue import DeferredMemoryQueue
from memory_deferred_worker import DeferredMemoryWorker
from memory_deferred_processor import DeferredMemoryProcessor as Processor

O='fictional_owner';C='fictional_character';PRIVATE='FICTIONAL_PRIVATE_A5C_927'
class E:
    dimension=2;model_id='fictional-parent-a5c'
    def embed(self,text):return [1.0,0.2]
class K:
    def __init__(self):self.fail=False;self.calls=[]
    def decide(self,text):
        self.calls.append(text)
        if self.fail:raise RuntimeError(PRIVATE)
        return {'key':'喜好','how':'category_only','needs_review':True}
class FailBackend(SQLiteVectorBackend):
    def __init__(self,s,e):super().__init__(s,e.dimension,e.model_id);self.fail=False;self.calls=0
    def upsert(self,*args,**kwargs):
        self.calls+=1
        if self.fail:raise RuntimeError(PRIVATE)
        return super().upsert(*args,**kwargs)

class Parent(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=MemoryStore(Path(self.tmp.name)/'fictional.sqlite3');self.e=E();self.backend=FailBackend(self.store,self.e);self.indexer=IndexWorker(self.store,self.e,self.backend);self.keyer=K();self.clock=100.0;self.workers=[]
    def tearDown(self):
        for w in self.workers:self.assertTrue(w.stop(timeout=2))
        self.store.close();self.tmp.cleanup()
    def make(self,**kw):
        self.assertIsNotNone(Processor,'A5c processor absent in accepted baseline (expected RED)')
        return Processor(self.store,O,C,kw.pop('keyer',self.keyer),kw.pop('indexer',self.indexer),**kw)
    def worker(self,processor):
        q=DeferredMemoryQueue(self.store,O,C,idle_seconds=1,clock=lambda:self.clock)
        w=DeferredMemoryWorker(q,processor,retry_seconds=1);self.workers.append(w);return q,w
    def search(self):return SQLiteSemanticSearcher(self.store,self.e,self.backend).search(O,C,'虛構星圖',limit=5,domain='real')
    def test_native_index_failure_retry_empty_new_ids_and_real_queue_ack(self):
        proc=self.make();q,w=self.worker(proc);text='我喜歡虛構星圖攝影';mid=w.capture(text);self.clock+=2;self.backend.fail=True
        self.assertEqual(w.process_once(),{'status':'retry','error_category':'index_failed'})
        self.assertEqual(q.status()['done_count'],0);self.assertEqual(q.status()['pending_count'],1)
        self.assertEqual(self.store.list_memories(O,C)['total'],1);self.assertEqual(len(self.store.pending_jobs(O,C)),1)
        self.backend.fail=False;self.clock+=2
        self.assertEqual(w.process_once(),{'status':'ok'});self.assertEqual(q.status()['done_count'],1)
        rows=self.store.list_memories(O,C)['items'];self.assertEqual(len(rows),1);self.assertEqual(rows[0]['source_ids'],[mid]);self.assertEqual(rows[0]['text'],text)
        self.assertEqual(self.keyer.calls,[text]);self.assertEqual(self.store.pending_jobs(O,C),[]);self.assertEqual(len(self.search()),1)
    def test_native_keyer_error_trace_not_ack_then_success(self):
        self.keyer.fail=True;proc=self.make();q,w=self.worker(proc);text='我喜歡虛構星圖模型';mid=w.capture(text);self.clock+=2
        r=w.process_once();self.assertEqual(r,{'status':'retry','error_category':'processor_failed'});self.assertNotIn(PRIVATE,json.dumps(r))
        self.assertEqual(q.status()['done_count'],0);self.assertEqual(self.store.list_memories(O,C)['total'],0)
        self.keyer.fail=False;self.clock+=2
        self.assertEqual(w.process_once(),{'status':'ok'});self.assertEqual(q.status()['done_count'],1)
        rows=self.store.list_memories(O,C)['items'];self.assertEqual(len(rows),1);self.assertEqual(rows[0]['source_ids'],[mid]);self.assertEqual(len(self.search()),1)
    def test_partial_extraction_is_indexed_but_not_successfully_finished(self):
        class Partial(K):
            failed=True
            def decide(self,text):
                if self.failed and '彗星' in text:raise RuntimeError(PRIVATE)
                return super().decide(text)
        k=Partial();proc=self.make(keyer=k);text='我喜歡虛構星圖攝影。我喜歡虛構彗星展覽。'
        self.assertEqual(proc(text),{'status':'failed','error_category':'processor_failed'})
        self.assertEqual(self.store.list_memories(O,C)['total'],1);self.assertEqual(self.store.pending_jobs(O,C),[]);self.assertEqual(len(self.search()),1)
        k.failed=False;self.assertEqual(proc(text),{'status':'ok'});rows=self.store.list_memories(O,C)['items'];self.assertEqual(len(rows),2);self.assertEqual(len({r['text'] for r in rows}),2);self.assertEqual(rows[0]['source_ids'],rows[1]['source_ids'])
    def report(self,**kw):
        r={'status':'ok','owner_id':O,'character_id':C,'failed_count':0,'pending_count':0,'pending_count_status':'exact','pending_count_exact':True};r.update(kw);return r
    def test_false_empty_report_cannot_ack_authoritative_pending(self):
        record_user_message_keyed(self.store,O,C,'我喜歡虛構星圖收藏',keyer=self.keyer);before=self.store.pending_jobs(O,C);self.assertEqual(len(before),1)
        calls=[]
        def lie(*args,**kw):calls.append((args,kw));return self.report()
        proc=self.make(indexer=SimpleNamespace(run_once=lie),max_index_batches=2)
        r=proc('晚安');self.assertEqual(r,{'status':'failed','error_category':'index_failed'});self.assertEqual(len(calls),2);self.assertEqual(self.store.pending_jobs(O,C),before)
    def test_malformed_nonexact_boolean_or_foreign_reports_are_not_success(self):
        bads=[None,{},self.report(pending_count=None),self.report(pending_count=False),self.report(failed_count=False),self.report(pending_count_exact=False,pending_count_status='unknown'),self.report(owner_id='fictional_other'),self.report(status='failed')]
        for report in bads:
            with self.subTest(report=report):
                proc=self.make(indexer=SimpleNamespace(run_once=lambda *a,_r=report,**k:_r))
                self.assertEqual(proc('晚安'),{'status':'failed','error_category':'index_failed'})
    def test_source_exclusion_and_foreign_jobs_remain_separate(self):
        foreign=record_user_message_keyed(self.store,O,'fictional_other','我喜歡虛構彗星模型',keyer=K());foreign_jobs=self.store.pending_jobs(O,'fictional_other');self.assertEqual(len(foreign_jobs),1)
        proc=self.make();text='我喜歡虛構星圖攝影';self.assertEqual(proc(text),{'status':'ok'});row=self.store.list_memories(O,C)['items'][0]
        self.assertTrue(self.store.delete_memory(O,C,row['memory_id']));calls=len(self.keyer.calls)
        self.assertEqual(proc(text),{'status':'ok'});self.assertEqual(len(self.keyer.calls),calls);self.assertEqual(self.store.list_memories(O,C)['total'],0);self.assertEqual(self.search(),[])
        self.assertEqual(self.store.pending_jobs(O,'fictional_other'),foreign_jobs);self.assertIsNotNone(self.store.get_memory(O,'fictional_other',foreign['memory_ids'][0]))
if __name__=='__main__':unittest.main(verbosity=2)
