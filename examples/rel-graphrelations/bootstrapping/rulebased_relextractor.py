# coding=utf-8
__author__ = 'hadyelsahar'

from sklearn.base import BaseEstimator, ClassifierMixin
import os
from corenlp.corenlpclient import *
from unicodedata import normalize
import codecs


# Predifine relations tags here for easy change
__RELATIONS__ = {
    "s_p": "s_p",
    "p_o": "p_o",
    "p_c": "p_c",
    "c_p": "c_p",
    "p_p": "p_p",
    "c_co": "c_co",
    "coref": "coref",
    "poss": "poss",
    "is-specialized-by": "is-specialized-by",
    "conj": "conj",
    "compound": "compound"
}

__DEPRELATIONS__ = {
    "compound": "compound",
    "aux": "aux",
    "neg": "neg",
    "auxpass": "auxpass",
    "mwe": "mwe",
    "name": "name",
    "cop": "cop",
    "nsubj": "nsubj",
    "nsubjpass": "nsubjpass",
    "nmod": "nmod",
    "nmod:tmod": "nmod:tmod",
    "nmod:npmod": "nmod:npmod",
    "nmod:poss": "nmod:poss",
    "case": "case",
    "ccomp": "ccomp",
    "dobj":"dobj",
    "iobj":"iobj",
    "amod": "amod",
    "advmod": "advmod",
    "nummod": "nummod",
    "acl": "acl"
}

class RuleBasedRelationExtractor(BaseEstimator, ClassifierMixin):

    def __init__(self, rawdir="./rawfiles/", outputdir="./output/"):

        self.relations_dict = __RELATIONS__
        self.dep_relations = __DEPRELATIONS__

        self.rawdir = rawdir
        self.outputdir = outputdir
        self.parser = CoreNlPClient()


    def fit(self, X, y):

        pass

    def predict(self, X):
        pass

    def bootstrap(self):
        total_c = len(os.listdir(self.rawdir))
        rawfilenames = os.listdir(self.rawdir)
        rawfilenames = ['db_1000.txt','0000-SAMPLE.txt']  #/ for testing
        for c, i in enumerate(rawfilenames):

            # try:
                # print i
                s = open(self.rawdir+i).read()

                s = s.replace("ä", "ae")

                # Getting dependency parse tree, NER and COREF
                parse = self.parser.annotate(s)

                # Generate new relations for the custom representations
                relations = self.extractrelations(s, parse)

                # Export the custom relations in a .ann file and write both .txt and .ann into the outdir
                self.save_in_brat_format(i.replace(".txt", ""), s, parse, relations)

                if c % 10 == 0 :
                    print "bootstrapped %d out of %d" % (c, total_c)

                # return relations
            # except:
            #     print "can't bootstrapp file %s " % i


    def extractrelations(self, s, parse):
        """
        :param s: raw sentence in text
        :param parse: parse class instance that contains the parsed results
        :return:
        """

        # Extract Compound Relations :
        relations = [{"in": [], "out":[]} for i in parse.tokens]

        ###################################
        # Extraction of compound relations
        ###################################
        for i, token in enumerate(parse.dep):
            for r in token["out"]:
                compound_dep_relations = [__DEPRELATIONS__["compound"],
                                          __DEPRELATIONS__["aux"],
                                          __DEPRELATIONS__["neg"],
                                          __DEPRELATIONS__["auxpass"],
                                          __DEPRELATIONS__["mwe"],
                                          __DEPRELATIONS__["name"]
                                          ]

                if r[0] in compound_dep_relations:
                    gov = i
                    depndnt = r[1]
                    relations[gov]["out"].append((__RELATIONS__["compound"], depndnt))
                    relations[depndnt]["in"].append((__RELATIONS__["compound"], gov))


        ######################################################################################
        # Extraction of Compound nouns, Dates or Named Entities from the Core NLP NER Tagger #
        ######################################################################################
        ner = parse.ner
        # a compound phrase should be one governor and one or many dependents
        # compound = {"gov": , "dep": []}
        compound = {"gov": None, "dep": []}

        for i, tag in enumerate(ner):

            if tag == 'O':
                pass
            else:
                # if tag == next tag (or not the last element) add to the stack as dependent of the compound relation
                if i+1 < len(ner) and tag == ner[i+1]:
                    compound["dep"].append(i)
                # if it's the last tagged ner, add it as the governor to all the words in the stack
                else:
                    # check if already there's a gov put it as dep if not then last word is the gov
                    compound["gov"] = i

                    # post process the compound, if only one word has relations with outside the compound
                    # then make it the new governor
                    minid = min(compound["dep"] + [compound["gov"]])
                    maxid = max(compound["dep"] + [compound["gov"]])

                    for s in compound["dep"]:
                        t = [t[1] for t in parse.dep[s]["out"] + parse.dep[s]["in"]]
                        # if  the word has any dependency relations ( in or out ) with a word outside of the compound
                        # then it's the new gov.
                        if len([outdeps for outdeps in t if outdeps < minid or outdeps > maxid]):
                            compound["dep"].remove(s)
                            compound["dep"].append(compound["gov"])
                            compound["gov"] = s

                    for s in compound["dep"]:
                        relations[compound["gov"]]["out"].append((__RELATIONS__["compound"], s))
                        relations[s]["in"].append((__RELATIONS__["compound"], compound["gov"]))
                    compound = {"gov": None, "dep": []}


        #############################################
        # From Dependency To Custom Representations #
        #############################################

        # Rule 13
        # Copular verb then acl
        # eg: Minecraft is a game company acquired by Microsoft
        # token ← acl ← gov2
        # gov2 →  cop
        # gov2 → nsubj → gov1
        # then : token → nsubj → gov1

        for i, token in enumerate(parse.dep):

            if parse.postags[i].startswith("VB"):

                for d_in in [r for r in token["in"] if r[0] == __DEPRELATIONS__["acl"]]:

                    gov2id = d_in[1]
                    gov2_dep_out_names = [r[0] for r in parse.dep[gov2id]["out"]]

                    if __DEPRELATIONS__["nsubj"] in gov2_dep_out_names and __DEPRELATIONS__["cop"] in gov2_dep_out_names:
                        for gov1 in [r for r in parse.dep[gov2id]["out"] if __DEPRELATIONS__["nsubj"] == r[0] ]:

                            gov1id = gov1[1]
                            depid = i

                            print "Rule 13 Triggered "

                            # add subject --> predicate (is) relations
                            relations[gov1id]["out"].append((__RELATIONS__["s_p"], depid))
                            relations[depid]["in"].append((__RELATIONS__["s_p"], gov1id))


        # Rule 0 : Addition of copular verbs are predicates:
        for i, token in enumerate(parse.ccdep):

            outr = [r[0] for r in token["out"]]
            outindx = [r[1] for r in token["out"]]

            if __DEPRELATIONS__["cop"] in outr:
                for subjid, subjrel in enumerate(outr):
                    if subjrel == __DEPRELATIONS__["nsubj"]:

                        subjid = outindx[subjid]
                        predid = outindx[outr.index(__DEPRELATIONS__["cop"])]
                        objid = i
                        # add subject --> predicate (is) relations
                        relations[subjid]["out"].append((__RELATIONS__["s_p"], predid))
                        relations[predid]["in"].append((__RELATIONS__["s_p"], subjid))
                        # add predicate --> object relations
                        relations[predid]["out"].append((__RELATIONS__["p_o"], objid))
                        relations[objid]["in"].append((__RELATIONS__["p_o"], predid))

        # Rule 2.0 : `ccomp` dep relation into P_P "Believe he ate his food"
        # Rule 2.1 : `ccomp` dep relation into P_P when there's copular file
        # (ie. not a predicate to have context relation)
        # todo : handle ccomp with copular verbs  e.g. : he believes he is good
        for i, token in enumerate(parse.dep):

            inr = [r[0] for r in token["in"]]
            inindex = [r[1] for r in token["in"]]

            outr = [r[0] for r in token["out"]]
            outindx = [r[1] for r in token["out"]]

            predrelation = __RELATIONS__["p_p"]

            if __DEPRELATIONS__["ccomp"] in inr:

                p1id = inindex[inr.index(__DEPRELATIONS__["ccomp"])]

                # if there's a copular verb attach the P_P to the copular verb
                # e.g. : people believe earth was flat
                if __DEPRELATIONS__["cop"] in outr:
                    p2id = outindx[outr.index(__DEPRELATIONS__["cop"])]

                else:
                    p2id = i

                # add p1 -> P_P -> p2  relation
                relations[p1id]["out"].append((predrelation, p2id))
                relations[p2id]["in"].append((predrelation, p1id))

        # Rule 3: nsubj | nsubjpass  --> s_p
        # not Rule 0:
        # print parse.ccdep
        for i, token in enumerate(parse.ccdep):

            outr = [r[0] for r in token["out"]]
            outindx = [r[1] for r in token["out"]]

            subjrels = {__DEPRELATIONS__["nsubjpass"], __DEPRELATIONS__["nsubj"]}

            if subjrels.intersection(set(outr)) and __DEPRELATIONS__["cop"] not in outr:

                rel = list(subjrels.intersection(set(outr)))[0]

                govid = outindx[outr.index(rel)]
                depid = i

                # add subject --> predicate (is) relations
                relations[govid]["out"].append((__RELATIONS__["s_p"], depid))
                relations[depid]["in"].append((__RELATIONS__["s_p"], govid))

        # Rule 4: Direct and indirect object :  iobj | dobj --> P_O
        for i, token in enumerate(parse.dep):

            outr = [r[0] for r in token["out"]]
            outindx = [r[1] for r in token["out"]]

            objrels = {__DEPRELATIONS__["dobj"], __DEPRELATIONS__["iobj"]}

            for rel in objrels.intersection(set(outr)):

                govid = i
                depid = outindx[outr.index(rel)]

                # add subject --> predicate (is) relations
                relations[govid]["out"].append((__RELATIONS__["p_o"], depid))
                relations[depid]["in"].append((__RELATIONS__["p_o"], govid))

        # Rule 1.0 : addition of context-prep from `case` dep relation
        # e.g. : he ate lunch at the restaurant
        # Rule 1.1 : addition of is-specialized-by if the nmod doesn't have s_p prop as output relation
        # e.g. : the president of russia
        # (ie. not a predicate to have context relation)
        for i, token in enumerate(parse.dep):

            inr = [r[0] for r in token["in"]]
            inindex = [r[1] for r in token["in"]]

            outr = [r[0] for r in token["out"]]
            outindx = [r[1] for r in token["out"]]

            mods_relations = [__DEPRELATIONS__["nmod:npmod"], __DEPRELATIONS__["nmod:npmod"], __DEPRELATIONS__["nmod"]]

            # gov0id -- r1 : p_c | is-specialized by  -->  gov1id ---> r2 : c_co --> depid
            for dep_rel in [r for r in token["in"] if r[0] in mods_relations]:

                govid = dep_rel[1]
                depid = i

                # P_C is the predicate is a (relation) otherwise is-specialized-by

                if __RELATIONS__["s_p"] in [r[0] for r in relations[govid]["in"]]:
                    predrelation = __RELATIONS__["p_c"]
                else:
                    predrelation = __RELATIONS__["is-specialized-by"]

                if __DEPRELATIONS__["case"] in outr:

                    prepid = outindx[outr.index(__DEPRELATIONS__["case"])]

                    # add subject --> predicate (is) relations
                    relations[govid]["out"].append((predrelation, prepid))
                    relations[prepid]["in"].append((predrelation, govid))

                    # add predicate --> object relations
                    relations[prepid]["out"].append((__RELATIONS__["c_co"], depid))
                    relations[depid]["in"].append((__RELATIONS__["c_co"], prepid))

                else:
                    # add subject --> predicate (is) relations
                    relations[govid]["out"].append((predrelation, depid))
                    relations[depid]["in"].append((predrelation, govid))


        # Rule 5:

        # Rule 6: advmod (without s_p) | amod ---> is-specialized-by
        # advmod (+ s_p ) --> p_c    e.g. : she ate her food happily
        for i, token in enumerate(parse.dep):

            outr = [r[0] for r in token["out"]]
            outindx = [r[1] for r in token["out"]]

            objrels = [__DEPRELATIONS__["amod"], __DEPRELATIONS__["advmod"], __DEPRELATIONS__["nummod"]]

            for rel in [r for r in token["out"] if r[0] in objrels]:

                govid = i
                depid = rel[1]

                # P_C is the predicate is a (relation) otherwise is-specialized-by
                if __RELATIONS__["s_p"] in [r[0] for r in relations[govid]["in"]]:
                    predrelation = __RELATIONS__["p_c"]
                else:
                    predrelation = __RELATIONS__["is-specialized-by"]

                # add subject --> predicate (is) relations
                relations[govid]["out"].append((predrelation, depid))
                relations[depid]["in"].append((predrelation, govid))


        # Rule 14: possessiveness
        for i, token in enumerate(parse.dep):

            # get the apostrophe 's
            case = [x for x in token["out"] if x[0] == __DEPRELATIONS__["case"] and parse.postags[x[1]] == "pos"][0]
            # get the dependent
            nmodposs = [x for x in token["in"] if x[0] == __DEPRELATIONS__["nmod:poss"]]


            for c in case:
                govid = i
                depid = c[1]

                relations[gov1id]["out"].append((__RELATIONS__["compounds"], depid))
                relations[depid]["in"].append((__RELATIONS__["compound"], govid))

            for poss in nmodposs:
                depid = i
                govid = poss[1]

                relations[gov1id]["out"].append((__RELATIONS__["poss"], depid))
                relations[depid]["in"].append((__RELATIONS__["poss"], govid))

        #############################################################
        # Removal of redundant relations and relations within phrases
        #############################################################

        relations = [{"in": list(set(i["in"])), "out":list(set(i["out"]))} for i in relations]

        for govid,r in enumerate(relations):

            cmp_rels_ids = [i[1] for i in r["out"] if i[0] == __RELATIONS__["compound"]]

            # keep only the "in compound relations" for the dep:
            # remove other relations between the gov and dependents if they have compound relation between them

            relations[govid]["out"] = [i for i in relations[govid]["out"]
                                       if i[0] == __RELATIONS__["compound"] or i[1] not in cmp_rels_ids]

            for depid in cmp_rels_ids:
                relations[depid]["in"] = [(__RELATIONS__["compound"], govid)]


        return relations

    def save_in_brat_format(self, fname, s, parse, relations):
        """
        method to omit bootstrapped relations in brat annotation file
        :param fname: input file name fname.txt fname.ann
        :param s: sentence
        :param parse: Parse Class output from CoreNLP library client
        :param relations: bootstrapped custom relations
        :return: create .txt file contains raw sentences and .ann file contais annotations
        """

        txtfile = open("%s%s.txt" % (self.outputdir, fname), 'w')
        txtfile.write(s)
        txtfile.close()


        ann = ""
        # addition of segments as tokens
        for i, token in enumerate(parse.all['sentences'][0]['tokens']):
            ann += "T%d\tsegment %d %d\t%s\n" % (i, token["characterOffsetBegin"], token["characterOffsetEnd"], token["word"])

        # additions of relations
        rid = 0
        for tokid, token in enumerate(relations):
            for r in token["out"]:
                ann += "R%d\t%s Arg1:T%d Arg2:T%d\n" % (rid, r[0], tokid, r[1])
                rid += 1

        annfile = open("%s%s.ann" % (self.outputdir, fname), 'w')
        annfile.write(ann.encode("utf-8"))
        annfile.close()










