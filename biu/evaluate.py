from __future__ import division
import glob
import codecs
from collections import defaultdict as dd
import numpy as np
from Levenshtein import distance

# def distance(str1, str2):
#     """Simple Levenshtein implementation for evalm."""
#     m = np.zeros([len(str2)+1, len(str1)+1])
#     for x in xrange(1, len(str2) + 1):
#         m[x][0] = m[x-1][0] + 1
#     for y in xrange(1, len(str1) + 1):
#         m[0][y] = m[0][y-1] + 1
#     for x in xrange(1, len(str2) + 1):
#         for y in xrange(1, len(str1) + 1):
#             if str1[y-1] == str2[x-1]:
#                 dg = 0
#             else:
#                 dg = 1
#             m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
#     return int(m[len(str2)][len(str1)])

def read_gold():
    """ read in the answers """
    gold = {}
    for filename in glob.iglob("gold/*"):
        if "test" not in filename:
            continue
        lang, task, _ = filename.split("/")[1].split("-")
        if lang not in gold:
            gold[lang] = {}
        if task not in gold[lang]:
            gold[lang][task] = {}

        with codecs.open(filename, 'rb', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                split = line.split("\t")
                key, answer = None, None
                if task == "task2":
                    key = tuple(split[1:3])
                    answer = split[3]
                else:
                    key = tuple(split[:2])
                    answer = split[2]
                gold[lang][task][key] = answer
    return gold
                
def read_file(fname, task):
    """ reads in the file """
    guesses = dd(list)
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            split = line.split("\t")
            if "alberta" in fname:
                split = split[:-1]
            key, answer = None, None
            if task == "task2":
                # columbia system transposed rows
                if "columbia" in fname:
                    tmp = split[0]
                    split[0] = split[1]
                    split[1] = tmp
                
                answer = split[-1]
                # colorado system had a bug in maltese
                if len(split) < 4:
                    key = tuple(split[:2])
                else:
                    key = tuple(split[1:3])
            else:
                key = tuple(split[:2])
                # handles input  correct file
                if len(split) < 3:
                    answer = ""
                else:
                    answer = split[2]
            guesses[key].append(answer)
            #if "maltese" in fname and "task2" in fname:
            #    print guesses[key]
            #    raw_input()
    guesses = dict(guesses)
    return guesses

def evaluate(gold, guesses):
    """ evaluates """
    # accuracy
    correct, total = 0, 0
    lev = 0
    rank = 0
    for key, lst in guesses.items():
        best = lst[0]
        # need to handle key error better
        # Columbia NYU-AD system has some issues with task 2
        # they plan to resubmit though
        if key in gold:
            if best == gold[key]:
                correct += 1
            lev += distance(unicode(best), unicode(gold[key]))
            r = 0.0
            try:
                 r = 1.0/(lst.index(gold[key])+1)
            except ValueError:
                 # 1.0 / oo = 0
                 r = 0.0
            rank += r
        total += 1
    return correct / total, lev / total, rank / total
                
def main():
    gold = read_gold()
    for filename in glob.iglob('submission*/**/*/*'):
        team, entry, track, rest = filename.split("/")
        lang, task, _ = rest.split("-")
        acc, lev, rank = evaluate(gold[lang][task], read_file(filename, task))
        print "\t".join(map(str, [team, entry, track, lang, task, acc, lev, rank]))

if __name__ == "__main__":
    main()
