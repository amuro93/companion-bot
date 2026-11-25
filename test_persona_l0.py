"""L0 回憶層按需載入的契約測試。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from companion_chat import ChatScope, CompanionChat, PERSONA_MAX
from persona_l0 import MAX_L0_CHARS, PersonaL0, parse_triggers


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class PersonaL0ParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_always_and_trigger_headers(self):
        write(self.dir, "a.md", "# 甲\n\n> load: always\n\n內容甲\n")
        write(self.dir, "b.md", "# 乙\n\n> load: trigger 生日 幾歲\n\n內容乙\n")
        l0 = PersonaL0(self.dir)
        self.assertEqual([d.name for d in l0.docs], ["a.md", "b.md"])
        self.assertEqual(l0.docs[0].load, "always")
        self.assertEqual(l0.docs[0].triggers, ())
        self.assertEqual(l0.docs[1].load, "trigger")
        self.assertEqual(l0.docs[1].triggers, ("生日", "幾歲"))
        self.assertEqual(l0.docs[1].title, "乙")

    def test_file_without_header_is_not_loaded(self):
        write(self.dir, "a.md", "# 甲\n\n沒有宣告\n")
        self.assertEqual(PersonaL0(self.dir).docs, [])

    def test_underscore_template_is_skipped(self):
        write(self.dir, "_TEMPLATE_x.md", "# 範本\n\n> load: always\n\nx\n")
        write(self.dir, "a.md", "# 甲\n\n> load: always\n\ny\n")
        self.assertEqual([d.name for d in PersonaL0(self.dir).docs], ["a.md"])

    def test_parse_triggers_dedupes_and_keeps_order(self):
        self.assertEqual(parse_triggers("生日，幾歲 生日、年次"), ("生日", "幾歲", "年次"))

    def test_invalid_dir_raises(self):
        with self.assertRaises(ValueError):
            PersonaL0(self.dir / "nope")
        with self.assertRaises(ValueError):
            PersonaL0(self.dir, max_chars=0)


class PersonaL0RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write(self.dir, "always.md", "# 關係\n\n> load: always\n\nALWAYS-BODY\n")
        write(self.dir, "trig.md", "# 生活面\n\n> load: trigger 生日 退伍\n\nTRIGGER-BODY\n")
        self.l0 = PersonaL0(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_always_doc_renders_every_turn(self):
        out = self.l0.render("今天天氣不錯")
        self.assertIn("ALWAYS-BODY", out)
        self.assertNotIn("TRIGGER-BODY", out)

    def test_trigger_doc_renders_only_on_match(self):
        out = self.l0.render("我生日快到了")
        self.assertIn("ALWAYS-BODY", out)
        self.assertIn("TRIGGER-BODY", out)

    def test_index_lists_docs_and_forbids_fabrication(self):
        out = self.l0.render("隨便聊")
        self.assertIn("【我記得的事（索引）】", out)
        self.assertIn("關係", out)
        self.assertIn("生活面", out)
        self.assertIn("生日", out)          # 觸發詞本身要看得到
        self.assertIn("不要憑空編造", out)

    def test_oversized_trigger_doc_is_skipped_with_a_note(self):
        write(self.dir, "big.md", "# 大檔\n\n> load: trigger 大\n\n" + "x" * (MAX_L0_CHARS + 1) + "\n")
        l0 = PersonaL0(self.dir)
        out = l0.render("大")
        self.assertIn("ALWAYS-BODY", out)
        self.assertIn("這次沒展開", out)
        self.assertNotIn("x" * 1000, out)


def update(text, user=11, kind="private"):
    return SimpleNamespace(effective_user=SimpleNamespace(id=user),
                           effective_chat=SimpleNamespace(id=user, type=kind),
                           message=SimpleNamespace(text=text))


class EmptyMemory:
    def search(self, token, query, limit, domain):
        return []


class StubPersonaL0:
    def __init__(self, text="", boom=False):
        self.text = text
        self.boom = boom
        self.seen = []

    def render(self, user_text):
        self.seen.append(user_text)
        if self.boom:
            raise RuntimeError("broken")
        return self.text


class ChatInjectionTests(unittest.TestCase):
    def make(self, persona="P", l0=None):
        prompts = []

        def llm(messages):
            prompts.append(messages)
            return "好"

        chat = CompanionChat({11: ChatScope("t" * 8, "owner", "char")},
                             EmptyMemory(), llm, persona, persona_l0=l0)
        return chat, prompts

    def test_l0_is_appended_to_the_system_prompt(self):
        l0 = StubPersonaL0("【L0-OK】")
        chat, prompts = self.make(l0=l0)
        chat.turn(update("我生日幾號"))
        self.assertEqual(l0.seen, ["我生日幾號"])
        system = prompts[0][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("P", system["content"])
        self.assertIn("【L0-OK】", system["content"])

    def test_broken_l0_never_breaks_the_turn(self):
        chat, prompts = self.make(l0=StubPersonaL0(boom=True))
        self.assertEqual(chat.turn(update("hi")), "好")
        self.assertNotIn("【L0-OK】", prompts[0][0]["content"])

    def test_persona_longer_than_the_old_2000_limit_is_accepted(self):
        chat, _ = self.make(persona="人" * 5000)
        self.assertEqual(chat.turn(update("hi")), "好")

    def test_persona_over_the_new_limit_still_rejected(self):
        with self.assertRaises(ValueError):
            self.make(persona="人" * (PERSONA_MAX + 1))

    def test_invalid_persona_l0_rejected(self):
        with self.assertRaises(ValueError):
            self.make(l0=SimpleNamespace(render="not callable"))


if __name__ == "__main__":
    unittest.main()
