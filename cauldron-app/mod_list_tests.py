import unittest
import json
import os
import pickle
from mods_list import ModsList, fresh_mods_list, save_mods_list_to_file, load_mods_list_from_file, suggest_mod, get_mods_list, push_mod, rank_mod, remove_mod  # Replace 'your_module' with the actual module name

class TestModsList(unittest.TestCase):

    def setUp(self):
        self.filename = 'test_mods_list.pkl'
        self.mod1 = {'id': 'mod1', 'priority': 5}
        self.mod2 = {'id': 'mod2', 'priority': 10}
        self.mod3 = {'id': 'mod3', 'priority': 3}
        self.mods_list = ModsList()
    
    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
    
    def test_suggest_mod(self):
        self.mods_list.suggest_mod(self.mod1)
        self.assertEqual(len(self.mods_list.queue), 1)
        self.mods_list.suggest_mod(self.mod2)
        self.assertEqual(len(self.mods_list.queue), 2)
    
    def test_get_mods_list(self):
        self.mods_list.suggest_mod(self.mod1)
        self.mods_list.suggest_mod(self.mod2)
        mods_list = self.mods_list.get_mods_list()
        self.assertEqual(len(mods_list), 2)
        self.assertEqual(mods_list[0]['id'], 'mod2')
    
    def test_push_mod(self):
        self.mods_list.suggest_mod(self.mod1)
        self.mods_list.suggest_mod(self.mod2)
        mod = self.mods_list.push_mod()
        self.assertEqual(mod['id'], 'mod1')
        self.assertEqual(len(self.mods_list.queue), 1)
    
    def test_rank_mod(self):
        self.mods_list.suggest_mod(self.mod1)
        self.mods_list.rank_mod('mod1', 1)
        self.assertEqual(self.mods_list.queue[0][0], -1)
    
    def test_remove_mod(self):
        self.mods_list.suggest_mod(self.mod1)
        self.mods_list.suggest_mod(self.mod2)
        self.mods_list.remove_mod('mod1')
        self.assertEqual(len(self.mods_list.queue), 1)
        self.assertEqual(self.mods_list.queue[0][1]['id'], 'mod2')
    
    def test_fresh_mods_list(self):
        fresh_mods_list(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        mods_list = load_mods_list_from_file(self.filename)
        self.assertIsInstance(mods_list, ModsList)
    
    def test_save_mods_list_to_file(self):
        self.mods_list.suggest_mod(self.mod1)
        save_mods_list_to_file(self.mods_list, self.filename)
        self.assertTrue(os.path.exists(self.filename))
    
    def test_load_mods_list_from_file(self):
        self.mods_list.suggest_mod(self.mod1)
        save_mods_list_to_file(self.mods_list, self.filename)
        loaded_mods_list = load_mods_list_from_file(self.filename)
        self.assertEqual(len(loaded_mods_list.queue), 1)
        self.assertEqual(loaded_mods_list.queue[0][1]['id'], 'mod1')
    
    def test_suggest_mod_tool(self):
        mod_json = json.dumps(self.mod1)
        result = suggest_mod(mod_json, self.filename)
        self.assertIn('mod1', result)
    
    def test_get_mods_list_tool(self):
        mods_list = ModsList()
        mods_list.suggest_mod(self.mod1)
        mods_list.suggest_mod(self.mod2)
        save_mods_list_to_file(mods_list, self.filename)
        result = get_mods_list(self.filename)
        self.assertIn('mod1', result)
        self.assertIn('mod2', result)
    
    def test_push_mod_tool(self):
        mods_list = ModsList()
        mods_list.suggest_mod(self.mod1)
        save_mods_list_to_file(mods_list, self.filename)
        result = push_mod(self.filename)
        self.assertIn('mod1', result)
    
    def test_rank_mod_tool(self):
        mods_list = ModsList()
        mods_list.suggest_mod(self.mod1)
        save_mods_list_to_file(mods_list, self.filename)
        result = rank_mod('mod1', 1, self.filename)
        self.assertIn('mod1', result)
    
    def test_remove_mod_tool(self):
        mods_list = ModsList()
        mods_list.suggest_mod(self.mod1)
        save_mods_list_to_file(mods_list, self.filename)
        result = remove_mod('mod1', self.filename)
        self.assertIn('successfully removed', result)

if __name__ == '__main__':
    unittest.main()
