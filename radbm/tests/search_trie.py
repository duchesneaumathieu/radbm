import unittest
from radbm.search.basic.trie import Trie, Node

class TestTrie(unittest.TestCase):
    def test_trie_insert(self):
        trie = Trie()
        
        #test taging root
        trie.insert(())
        self.assertFalse(trie.root) #no child
        self.assertTrue(trie.root.tag)
        
        #test adding one word
        trie = Trie()
        trie.insert('helicopter')
        self.assertIn(tuple('helicopter'), trie.root)
        self.assertEqual(trie.root[tuple('helicopter')].prefix, tuple('helicopter'))
        
        #adding hello, should create a splitting branch "hel" with two branches "lo" and "icopter" .
        #(''->helicopter) ==> (''->hel->(lo, icopter))
        trie.insert('hello')
        self.assertEqual(len(trie.root), 1)
        self.assertIn(tuple('hel'), trie.root)
        hel = trie.root[tuple('hel')]
        self.assertEqual(len(hel), 2)
        self.assertIn(tuple('lo'), hel)
        self.assertIn(tuple('icopter'), hel)
        self.assertFalse(trie.root.tag)
        self.assertFalse(hel.tag)
        self.assertTrue(hel[tuple('lo')].tag)
        self.assertTrue(hel[tuple('icopter')].tag)
        self.assertEqual(hel.prefix, tuple('hel'))
        self.assertEqual(hel[tuple('lo')].prefix, tuple('hello'))
        self.assertEqual(hel[tuple('icopter')].prefix, tuple('helicopter'))
        
        #adding "hel" should only affect hel.tag -> True
        trie.insert('hel')
        self.assertTrue(hel.tag) #hel should be the same no need to `hel = trie.root[tuple('hel')`
        
        #make sure every thing else is the same.
        self.assertEqual(len(trie.root), 1)
        self.assertIn(tuple('hel'), trie.root)
        self.assertEqual(len(hel), 2)
        self.assertIn(tuple('lo'), hel)
        self.assertIn(tuple('icopter'), hel)
        self.assertFalse(trie.root.tag)
        self.assertTrue(hel[tuple('lo')].tag)
        self.assertTrue(hel[tuple('icopter')].tag)
        self.assertEqual(hel.prefix, tuple('hel'))
        self.assertEqual(hel[tuple('lo')].prefix, tuple('hello'))
        self.assertEqual(hel[tuple('icopter')].prefix, tuple('helicopter'))
        
        #intermediate insert (hel->helicopter) ==> (hel->helico->helicopter)
        trie.insert('helico')
        self.assertEqual(len(hel), 2)
        self.assertIn(tuple('lo'), hel)
        self.assertIn(tuple('ico'), hel)
        helico = hel[tuple('ico')]
        self.assertTrue(helico.tag)
        self.assertEqual(helico.prefix, tuple('helico'))
        self.assertEqual(len(helico), 1)
        self.assertIn(tuple('pter'), helico)
        self.assertEqual(helico[tuple('pter')].prefix, tuple('helicopter'))
        self.assertTrue(helico[tuple('pter')].tag)
        
    def test_trie_node_setitem_errors(self):
        node = Node()
        with self.assertRaises(ValueError):
            node[2] = Node() #key not a tuple!
        
        with self.assertRaises(ValueError):
            node[(2,)] = 'not a node' #value not a node!
            
        trie = Trie()
        trie.insert('hello')
        trie.root[tuple('helicopter')] = Node(tag=True) #hel does not exist!
        #hello and helicopter are brothers and share the same prefix!!!
        with self.assertRaises(RuntimeError):
            trie.insert('hospital') #hospital has a common prefix (h) with both hello and helicopter
            #the insert method will be confused (it should not happen) and raise an error.
            
        with self.assertRaises(NotImplementedError):
            trie.search('a query')
            
    def test_trie_runtest(self):
        trie = Trie()
        trie.insert('test')
        repr(trie)
        trie.graph()