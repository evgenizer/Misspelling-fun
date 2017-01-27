#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:54:43 2017

@author: evgeny sorkin

A simple spellchecker: an adoption
       to an Object Oriented design of P Norvig's 
       http://norvig.com/spell-correct.html 

"""
from collections import Counter
import copy
class NSpell(object):
    """class that suggest misspelling corrections """
       
    def __init__(self, WORDS, word=''):
        if not isinstance(WORDS, Counter):
            WORDS  = Counter(copy.deepcopy(WORDS))
        self.__WORDS = WORDS
        self.word = word
        
    def P(self, word):
        """Probability of `word`."""
        try:
            return self.__WORDS[word] / sum(self.__WORDS.values())
        except:
            return 1.0

    def correction(self): 
        """Most probable spelling correction for word."""
        return max(self.candidates(), key=self.P)

    def candidates(self): 
        """Generate possible spelling corrections for word."""
        return (self.known([self.word]) or self.known(self.edits1(self.word)) 
                or self.known(self.edits2(self.word)) or [self.word])

    def known(self,words): 
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.__WORDS)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))   