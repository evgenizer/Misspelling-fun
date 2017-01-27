#!/usr/bin/python
"""
Created on Mon Jan 23 06:44:44 2017

@author: evgeny sorkin
@email : evg.sorkin@gmail.com
"""
import os
import re
import string
import collections


_NORM_REGEX = re.compile('([a-z])([A-Z][a-z])')
_WORD_REGEX = re.compile('[\s_0-9\W]+', flags=re.UNICODE)

##---------------------        IO    ---------------------------
class FolderStructure(object):
    """ Reads the folder structure to analyze"""

    def __init__(self, rootFolder ='.'):
        """
        reads foder structure into a dictionary
        @param: rootFoder, str - root Folder
        """
        self.__file_tree = collections.defaultdict()
        for root, dirs, files in os.walk(rootFolder):
            file_list=[]
            
            for filename in files:
                if filename.endswith('.txt'):
                    file_list.append(os.path.join(root, filename)) 
                    
            if os.path.basename(rootFolder) != os.path.basename(root):                
                self.__file_tree[str(os.path.basename(root))] = file_list
                                 
    def get_files(self): return self.__file_tree

 
def read_file(fn=''):
    """
    Reads text in the file named fn into a list, 
    do word normalization and cleaning
    """
    words = []
    try:
        fl = open(fn, 'r')
        words = [normalize(word) for line in fl for 
                     word in split_words(line) if len(word)>0]    
    except IOError:
        pass
#    if len(words) == 0: 
#        print("Warning: file {} not found, or empty...".format(fn) )
    return words      
    
def flatten(l, ltypes=(list, tuple)):
    """
    Function used to create a one dimensional list from a list of
    lists or single items.

    """
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)   
    
##----------------------   text analysis    ---------------------

def normalize(word):
  """Return word with symbols stripped from its ends."""
  
  return word.strip(string.punctuation)

def split_words(line):
  """Return the list of words contained in a line (string of words) ."""
  # Normalize any camel cased words first
  line = _NORM_REGEX.sub(r'\1 \2', line)
  return [normalize(w) for w in _WORD_REGEX.split(line)]
 

        
def main():
    """
    """
    import random
    import sys
    import classifier as classifier
    train_root = '../training'
    test_root = '../test'
    wordlist ='./wordlist.txt'
    
    task = int(sys.argv[1])    
    correct_errors= True
    
    print ('\n Performing task {}... Do misspelling correction? [{}] \n'.format(task,
           'Yes' if correct_errors else 'No'))
   
    train_files_dict = FolderStructure(train_root).get_files()
    test_files_dict = FolderStructure(test_root).get_files()
    
    train_files_list = flatten([v for k,v in train_files_dict.items()])
    test_files_list = flatten([v for k,v in test_files_dict.items()])
    
    WORDS = read_file(wordlist) if task==1 else None # English words, spelled correctly
    
    clf = classifier.Misspellings(train_files_list, WORDS)
    clf.fit()
    
    
    #for training purposes and cross_validation runs, e.g. to construct a learning curv
    #test_subsample = random.sample(test_files_list, 100)
    test_subsample = test_files_list
    #acc_cv = clf.cross_validate(train_files_list, cv_split = 0.8 )
            
    acc = clf.predict(test_subsample, bCV = False, bCorrect = correct_errors ) # change bCorrect = False if no corrections are desired
    print ('Average accuracy of the test data classification is {:.3f}%'.format(100*acc))
    
    print('Basic validate classifier on a random subset of test_files')
    if not clf.consistancy(random.sample(test_files_list, 20)):
        print('Invalid!!!')
    
 
#    print('Average accuracy of misspellings predictions given the vocabulary is {:.3f}%'.
#              format(100*clf.accuracy(test_subsample)))

if __name__ == "__main__":
    main() 
          