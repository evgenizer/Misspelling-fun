#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:45:40 2017

@author: evgeny sorkin
@email : evg.sorkin@gmail.com

The classifier, that does the sorting. 
Clustering functions used for Task 2 are included also



"""
import os
import numpy as np
from timeit import default_timer as timer
import collections
import Levenshtein 
from solution_evg import read_file
import nspell

MAX_WORDS_CLUSTERING = 10000  # the maximal number of words to be used in clustering algorithm
                             # if it is None, all words in training data will be used

MISSPELLINGS_FN = './misspellings.txt'
ext_e = 'error'    #extention given to errors-files
ext_a = 'accuracy' #extention given to accuracy-files
ext_c = 'correct'  #extention given to correction-files


def variants(w):
    """
    A simple speculative routine to add/remove some common word endings
    @param 
    # this simple addition improved classification 
    """
    ret = [w]
    if w[-1] == 's' : ret.append(w[:-1])
    else : ret.append(w+'s')
    if len(w)>1 and w[-2:] == 'ed': ret.append(w[:-2])     
    if len(w)>1 and w[-2:] != 'ed' and w[-1] != 'e': ret.append(w+'ed')
    if len(w)>1 and w[-2:] != 'ed' and w[-1] == 'e': ret.append(w+'d')
    
    return ret

def clusters(the_list):
    """ 
    Returns clusters of similar words in the list, Using Levenshtein distance, 
    We use sciit-learn affinityPropagation clustering algorithm to detect clusters
    @param the_list : list of words
    @note: unfortunately clustering algorithm is quadratic in len(the_list, so it is slow!
    """
    import sklearn.cluster
    words = np.asarray(the_list) #So that indexing with a list will work
    lev_similarity = -1*np.array([[np.sqrt(Levenshtein.distance(w1,w2)) for w1 in words] 
                                       for w2 in words])

    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", 
                                                  damping=0.5, 
                                                  max_iter=200)
                                                  #preference = -200)
    affprop.fit(lev_similarity)
    clusters_dict = collections.defaultdict();
    try:
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
            clusters_dict[exemplar] =cluster
    except Exception as e:
        print("Cannot create clusters in data due to {}. Exciting...".format(type(e)))
    return clusters_dict
    
def mean_dissimilarity(word, the_list):
    """ average dissimilarity of a word from a list of words"""
    num_comps = 0
    running_sum = 0
    for item in the_list:
        running_sum += Levenshtein.distance(word, item)
        num_comps += 1
    return float(running_sum)/float(num_comps)   

def get_winner(lstrings):
    """ find a string in lstrings with the smallest mean dissimialrity e.g. Levenshtein distance"""
    ltuples = []
    for item in lstrings:
        the_mean = mean_dissimilarity(item, lstrings)
        ltuples.append((the_mean, item))
    return sorted(ltuples)[0][1]       
## =========================================    main classifier   =============
class Misspellings(object):
    """ class the we use to train and test our models"""
    def __init__(self, files = None, WORDS= None, misspellings_file = None):
        """ """
        self.__files = []   
        if files:
            self.add(files)
                                            
        self.__WORDS = list(set([w.lower() for w in WORDS]))    # vocabulary, lower case
        self.__misspellings = [] # this is the "model" that we train 
        self.__misspellings_CV = [] # this is the "model" that we train when doing cross-validation
        
        if misspellings_file:
            self.__misspellings_file = misspellings_file
            self.__misspellings = read_file(misspellings_file)
        else:
            self.__misspellings_file = MISSPELLINGS_FN
        
    def add(self,files):
        """adds files to check"""
        
        if isinstance(files,str):
            self.__files.append(files)
        elif isinstance(files,list):
            self.__files.extend(files)
        else:
            pass
        
    def get_misspellings(self, bCV = False): 
        return self.__misspellings if not bCV else self.__misspellings_CV
    def get_vocabulary(self): return self.__WORDS
    def get_training_files(self): return self.__files
    
  
    def _create_misspellings(self, files_train = None, bCV = False):
        """
        This method creates the misspellings list, and saves it file.
        2 cases are treated:
            1. If vocabulary is provided, words are compared with it
            2. Otherwise we use unsupervised learning to learn correct misspelling from 
               train documents provided. We go through all the words and compare Levenshtein 
               distance between them, the word with minimal distance is considered correctly spelled.                
       @param files_train list of filename used for training, is not given uses self.__filenames 
       @param bCV: bool, true/false, whether it is a cross-validation run
        """
        start = timer()
        totw=0
        allerrors = []
        if None is files_train :
            files = self.__files
        else:
            files = files_train
        numfiles = len(files)
        words=[]
        if self.__WORDS: # task 1
            for i,fn in enumerate(files):
                if i % int(round(numfiles/10.0)) ==0  or i == numfiles-1: 
                    progress(i, numfiles-1, status = 'reading data')
                try:
                    words.extend(read_file(fn))
                except IOError:
                    continue
            print('\n Calculating misspellings...\n')
            totw = len(words)
            doc_lower = map(lambda x: x.lower(),list(set(words)))
            words_var =  map(variants, doc_lower)
            mask = map( lambda x: len(set(self.__WORDS).intersection(x)) ==0 , words_var)
            allerrors = [ w for i,w in enumerate(doc_lower) if mask[i] ]
                #allerrors.extend(list(errors) )
            
        else: # task 2 
            if None is self.__WORDS: self.__WORDS = []
            for fn in files:
                words.extend(list(set(read_file(fn))))
            
            words_set = list(set(map(lambda x: x.lower() ,set(words))  )) 
            totw = len(words_set)
            num_words =totw if MAX_WORDS_CLUSTERING is None else MAX_WORDS_CLUSTERING
            wclusters = clusters(words_set[:num_words]) #detect clusters
            totw = len(words_set[:num_words])
            errors=[]
            for _,cl in wclusters.items():
                cll = list(cl)
                good = get_winner(cll)
                cll.remove(good)
                self.__WORDS.extend(variants(good) )
                errors.extend(cll)

                allerrors.extend(list(errors))      
                                
        allerrors = list(set(allerrors))   
        if not bCV: 
            if not self.__misspellings:
                self.__misspellings=[]
            self.__misspellings.extend(list(allerrors)) 
            self.__misspellings=list(set(self.__misspellings))    
            
            #save to disk: create misspellings file
            with open(self.__misspellings_file, 'w') as mf:
                try:
                    for w in self.__misspellings: 
                        mf.write(w+'\n')  
                except Exception as e:
                    print('Failed to create misspellings file {} due to'.
                    format(mf,type(e)))
                    
            # validate        
            if not self._valid():
                print('Invalid, something went wrong!!')
        else:
            self.__misspellings_CV = allerrors
        end = timer()
        print('Training of the model took {} s at the rate {:.1f} words/s'.format(end-start, totw/(end-start)))
        return self.get_misspellings(bCV)
        
    def fit(self, files = None, bCV = False):
        """ 
        creates the model, ie the misspellings list and file
        @param bCV: bool, true/false, whether it is a cross-validation run
        """
        ret = []
        print('Fitting a model...')           
        if not self.get_misspellings(bCV):
            ret = self._create_misspellings(files, bCV)
        return ret    
    
    def predict(self, files, bCV = False, bCorrect = False):
        """ 
        generates .error files for all files 
        @param files: list of (str) filename
        @param bCV: bool, if cross-validation
        @param bCorrect: bool, suggest corrections - true/false
        """
        start = timer()
        print('Running the prediction on test/cv data...')
        if not self.get_misspellings(bCV):
            print('Has to train first!')
            return 
        totw=0 
        acc = []
        for i,fn in enumerate(files):
            if i % 2 ==0  or i == len(files)-1 : progress(i, len(files)-1, status = 'Predicting ')
            errors, misclassified, dw = self._calc_errors_doc(fn, bCV)
            acc.append(self._accuracy_score(errors,misclassified))
            totw += dw
            if len(errors)>0 and not bCV:
                self._save_errors(fn, errors,  misclassified) # create .error file
                if bCorrect:
                    self._save_corrections(fn, errors) # create .correct file    

        end = timer()
        print('\n Predicting of the model took {} s at rate {:.1f} words/s'.format(end-start, totw/(end-start)))
        return np.mean(acc) if len(acc)>0  else 0.0 
    
    def _calc_errors_doc(self, fn, bCV=False):
        """
        calculates misspelled words, missclassified words and
        the total number of words in a document
        @param fn: str, filename 
        """
        errors, misclassified, totw = [],[],0
        try:
            #import pdb; pdb.set_trace()
            doc = list(set(read_file(fn)))
            doc_lower = map(lambda x: x.lower(), doc)
            totw = len(doc)
            errors = list(set(self.get_misspellings(bCV)).intersection(doc_lower))
            rest =   list(set(doc_lower)-set(errors))  
            rest_var = map(variants, rest)
            mask = map( lambda x: len(set(self.__WORDS).intersection(x)) ==0 , rest_var)
            misclassified = [ w for i,w in enumerate(rest) if mask[i] ]
        except IOError:
            pass
        return errors, misclassified, totw

    def cross_validate(self, files, cv_split = 0.8):
        """
        Runs a cross-validation after splitting data into train and cv test. 
        @param files: list of (str) filename
        @param bC: bool, if cross-validation
        
        @note this is a place-holder at the moment
        """
        import random
        import copy
        if cv_split <=0 or cv_split >=1:
            print ('wrong cv_split factor,has to be between 0 and 1')
            return 0
        fs = copy.copy(files)
        l = len(fs)
        tt = int(cv_split*l)
        random.seed = 1234
        random.shuffle(fs)
        train_fs = fs[:tt]
        cv_fc = fs[tt:]
        self.__misspellings_CV = []
        __ = self.fit(train_fs, bCV=True)
        acc = self.predict(cv_fc, bCV = True)
        return acc
    def _accuracy_score(self,errors,misclassified):
        """ a definition for acuracy score"""
        return float(len(errors))/float(len(misclassified)+len(errors))
        

    def accuracy(self,files):
        """ simple accuracy score, percent of correct misspelling predictions
        The misspelling is classified as correct when a word is not in WORDS
        and it is in misspellings. if a word is not in misspellings 
        @param files: list of (str) test filenames
        """
    
        if not os.path.isfile(self.__misspellings_file):
            print('Has to train first!')
            return 
        acc =[] 
          
        for fn in files:
            filename, file_extension = os.path.splitext(fn)
            f = os.path.join(filename+'.'+ext_a) 
            try:
                acc.extend([float(w.strip('\n')) for w in open(f, 'r') if len(w)>0] )   
            except IOError:
                pass
            
        return np.mean(acc) if len(acc)>0 else 0.0        
                      
    def _valid(self):
        """ 
        Verifies that none of the words in self.__misspellings are in 
        self.__WORDS (provided that is given)
        @return: true/false
        """
        if not os.path.isfile(self.__misspellings_file):
            print('Has to train first!')
            return False
        return len(self.__misspellings + self.__WORDS) == \
                len(set(self.__misspellings + self.__WORDS))
       
        
    def consistancy(self,files):
        """
        Validate predictions:
        runs a consistancy check that all words in files classified as error are indeed 
        in self.__misspellings.
        @param files: list of (str) filename
        @return: true/false 
        """
        if not os.path.isfile(self.__misspellings_file):
            print('Has to train first!')
            return False
            
        for fn in files:
            filename, file_extension = os.path.splitext(fn)
            f = os.path.join(filename+'.'+ext_e) 
            try:
                for w in read_file(f): 
                    word = w.lower() 
                    if not word in self.__misspellings : return False 
            except IOError:
                continue   
        return True    
         
    def _save_errors(self,fn, errors,  misclassified):
        """
        saves misspellings file
        @param fn: str, filename
        @param errors: list
     
        @return : true/false upon success
        """
        filename, file_extension = os.path.splitext(fn)
        errorfn = os.path.join(filename+'.'+ext_e)
        accuracyfn = os.path.join(filename+'.'+ext_a)
        with open(accuracyfn, 'w') as f:
            f.write(str(float(len(errors))/float(len(misclassified)+len(errors)))+'\n')
        with open(errorfn, 'w') as ef:
            try:
                for w in errors: 
                    ef.write(w+'\n')
            except Exception as e:
                    print('Failed to save errors file {} due to {}'.
                             format(errorfn,type(e)))
                    return False 
        return True 
    def _save_corrections(self,fn, errors):
        """
        saves corrections file
        @param fn: str, filename
        @param errors: list
     
        @return : true/false upon success
        """
        
        filename, file_extension = os.path.splitext(fn)
        cfn = os.path.join(filename+'.'+ext_c)
        with open(cfn, 'w') as f:
            try:
                for w in errors: 
                    sc = nspell.NSpell(self.__WORDS,w)
                    corr= sc.correction()
                    corr_str  = corr if corr else "?"
                    f.write('{}, {} \n'.format(w,str(corr_str)))
            except Exception as e:
                    print('Failed to save correction file {} due to {}'.
                             format(cfn,type(e)))
                    import pdb;pdb.set_trace()
                    return False 
        return True     
## --------------------------------- some utils
def progress(count, total, status=''):
    import sys
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r')
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

         