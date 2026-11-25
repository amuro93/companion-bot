"""Parent-owned A-5c validation and native concurrency regressions; toy inference."""
import threading
import unittest
from unittest.mock import patch
import memory_deferred_processor as product
import test_memory_deferred_processor as native

class Boundary(unittest.TestCase):
    setUp=native.Parent.setUp
    tearDown=native.Parent.tearDown
    make=native.Parent.make
    worker=native.Parent.worker
    def test_invalid_configuration_rejected(self):
        args=[self.store,native.O,native.C,self.keyer,self.indexer]
        for position in (1,2):
            for value in (None,'',' ','x\n','x\x00'):
                changed=args.copy();changed[position]=value
                with self.subTest(position=position,value=repr(value)):
                    with self.assertRaises(ValueError):product.DeferredMemoryProcessor(*changed)
        for name in ('max_memories','max_index_batches'):
            for bad in (0,True,1.5,'2',None,101):
                with self.subTest(name=name,bad=bad):
                    with self.assertRaises((ValueError,TypeError)):self.make(**{name:bad})
        for position in (0,3,4):
            changed=args.copy();changed[position]=None
            with self.assertRaises((ValueError,TypeError)):product.DeferredMemoryProcessor(*changed)
    def test_refused_and_prefilter_rejected_are_legitimate_empty_success(self):
        class No:
            calls=0
            def decide(self,text):self.calls+=1;return {'key':None,'how':'no'}
        keyer=No();proc=self.make(keyer=keyer)
        self.assertEqual(proc('我喜歡虛構星圖攝影'),{'status':'ok'});self.assertEqual(keyer.calls,1)
        self.assertEqual(proc('晚安'),{'status':'ok'});self.assertEqual(keyer.calls,1)
        self.assertEqual(self.store.list_memories(native.O,native.C)['total'],0)
    def test_native_bounded_index_drain_leaves_pending_until_retry(self):
        for i in range(101):native.record_user_message_keyed(self.store,native.O,native.C,f'我喜歡虛構星圖收藏{i:03d}',keyer=self.keyer)
        self.assertEqual(len(self.store.pending_jobs(native.O,native.C,limit=200)),101)
        proc=self.make(max_index_batches=1)
        self.assertEqual(proc('晚安'),{'status':'failed','error_category':'index_failed'})
        self.assertEqual(len(self.store.pending_jobs(native.O,native.C,limit=200)),1)
        self.assertEqual(proc('晚安'),{'status':'ok'});self.assertEqual(self.store.pending_jobs(native.O,native.C),[])
        self.assertEqual(self.store.list_memories(native.O,native.C)['total'],101)
    def test_real_capture_during_keyer_inflight_not_blocked_by_database_transaction(self):
        entered=threading.Event();release=threading.Event();captured=threading.Event();errors=[];runs=[]
        class Blocking(native.K):
            def decide(self,text):
                entered.set()
                if not release.wait(10):raise RuntimeError('fictional blocked keyer timeout')
                return super().decide(text)
        proc=self.make(keyer=Blocking());q,w=self.worker(proc);w.capture('我喜歡虛構星圖攝影');self.clock+=2
        def process():
            try:runs.append(w.process_once())
            except BaseException as exc:errors.append(type(exc).__name__)
        def capture():
            try:w.capture('我喜歡虛構宇宙舞蹈')
            except BaseException as exc:errors.append(type(exc).__name__)
            finally:captured.set()
        hook=threading.Thread(target=process);other=threading.Thread(target=capture)
        hook.start()
        try:
            self.assertTrue(entered.wait(3));other.start();self.assertTrue(captured.wait(3));self.assertTrue(hook.is_alive());self.assertEqual(errors,[])
        finally:
            release.set();hook.join(3)
            if other.ident is not None:other.join(3)
        self.assertFalse(hook.is_alive());self.assertFalse(other.is_alive());self.assertEqual(runs,[{'status':'ok'}]);self.assertEqual(q.status()['done_count'],1);self.assertEqual(q.status()['pending_count'],1)
    def test_nested_error_and_malformed_recorder_results_cannot_be_acknowledged(self):
        proc=self.make();base={'message_id':'fictional_source','memory_ids':[],'results':[],'trace':[]}
        outcomes=[None,{**base,'trace':None},{**base,'memory_ids':None},{**base,'trace':[{'decision_trace':[{'how':'error'}]}]}]
        for result in outcomes:
            with self.subTest(result=result):
                with patch.object(product,'record_user_message_keyed',return_value=result):self.assertEqual(proc('fictional'),{'status':'failed','error_category':'processor_failed'})
        cyclic={};cyclic['decision_trace']=[cyclic];cyclic['how']='error'
        with patch.object(product,'record_user_message_keyed',return_value={**base,'trace':[cyclic]}):self.assertEqual(proc('fictional'),{'status':'failed','error_category':'processor_failed'})
if __name__=='__main__':unittest.main()
