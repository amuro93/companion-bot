import ast,inspect,unittest
import bootstrap_test_bot as bot
class EntryInvariants(unittest.TestCase):
 def test_local_factory_is_called_by_actual_entry(self):
  names={n.func.id for n in ast.walk(ast.parse(inspect.getsource(bot.run))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
  self.assertIn('build_local_memory_runtime',names)
 def test_retired_services_absent_from_actual_entry(self):
  names={n.id for n in ast.walk(ast.parse(inspect.getsource(bot.run))) if isinstance(n,ast.Name)}
  self.assertFalse(names & {'QdrantHTTPBackend','create_server','LoopbackMemoryClient','DecisionPrefilter','DECISION_KEY_FILE','record_user_message_keyed'})
if __name__=='__main__':unittest.main(verbosity=2)
