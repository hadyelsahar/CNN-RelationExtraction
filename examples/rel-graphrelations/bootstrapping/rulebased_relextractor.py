# coding=utf-8
__author__ = 'hadyelsahar'

from sklearn.base import BaseEstimator, ClassifierMixin
import os
from corenlp.corenlpclient import *

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
    "case": "case",
    "ccomp": "ccomp",
    "dobj":"dobj",
    "iobj":"iobj",
    "amod": "amod",
    "advmod": "advmod",
    "nummod": "nummod"
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
        for c, i in enumerate(os.listdir(self.rawdir)):

            try:
                # print i
                s = open(self.rawdir+i).read()

                # Getting dependency parse tree, NER and COREF
                parse = self.parser.annotate(s)

                # Generate new relations for the custom representations
                relations = self.extractrelations(s, parse)

                # Export the custom relations in a .ann file and write both .txt and .ann into the outdir
                self.save_in_brat_format(i.replace(".txt", ""), s, parse, relations)

                if c % 10 == 0 :
                    print "bootstrapped %d out of %d" % (c, total_c)

                # return relations
            except:
                print "can't bootstrapp file %s " % i


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
        for i, rels in enumerate(parse.dep):
            for r in rels["out"]:
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
        stack = []
        for i, tag in enumerate(ner):
            if tag == 'O':
                pass
            else:
                # if tag == next tag (or not the last element) add to the stack as dependent of the compound relation
                if i+1 < len(ner) and tag == ner[i+1]:
                    stack.append(i)

                # if it's the last tagged ner, add it as the governor to all the words in the stack
                else:
                    for s in stack:
                        relations[i]["out"].append((__RELATIONS__["compound"], s))
                        relations[s]["in"].append((__RELATIONS__["compound"], i))
                    stack = []


        ##################
        # From Dependency#
        ##################

        # Rule 0 : Addition of copular verbs are predicates:
        for i, rels in enumerate(parse.dep):

            outr = [r[0] for r in rels["out"]]
            outindx = [r[1] for r in rels["out"]]

            if __DEPRELATIONS__["nsubj"] in outr and __DEPRELATIONS__["cop"] in outr:

                subjid = outindx[outr.index(__DEPRELATIONS__["nsubj"])]
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
        for i, rels in enumerate(parse.dep):

            inr = [r[0] for r in rels["in"]]
            inindex = [r[1] for r in rels["in"]]

            outr = [r[0] for r in rels["out"]]
            outindx = [r[1] for r in rels["out"]]

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
        for i, rels in enumerate(parse.dep):

            outr = [r[0] for r in rels["out"]]
            outindx = [r[1] for r in rels["out"]]

            subjrels = {__DEPRELATIONS__["nsubjpass"], __DEPRELATIONS__["nsubj"]}

            if subjrels.intersection(set(outr)) and __DEPRELATIONS__["cop"] not in outr:

                rel = list(subjrels.intersection(set(outr)))[0]

                govid = outindx[outr.index(rel)]
                depid = i

                # add subject --> predicate (is) relations
                relations[govid]["out"].append((__RELATIONS__["s_p"], depid))
                relations[depid]["in"].append((__RELATIONS__["s_p"], govid))

        # Rule 4: Direct and indirect object :  iobj | dobj --> P_O
        for i, rels in enumerate(parse.dep):

            outr = [r[0] for r in rels["out"]]
            outindx = [r[1] for r in rels["out"]]

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
        for i, rels in enumerate(parse.dep):

            inr = [r[0] for r in rels["in"]]
            inindex = [r[1] for r in rels["in"]]

            outr = [r[0] for r in rels["out"]]
            outindx = [r[1] for r in rels["out"]]

            mods_relations = {__DEPRELATIONS__["nmod:npmod"], __DEPRELATIONS__["nmod:npmod"], __DEPRELATIONS__["nmod"]}

            # gov0id -- r1 : p_c | is-specialized by  -->  gov1id --- r2 : c_co--> depid
            for dep_rel in mods_relations.intersection(set(inr)):

                govid = inindex[inr.index(dep_rel)]
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

                else :
                    # add subject --> predicate (is) relations
                    relations[govid]["out"].append((predrelation, depid))
                    relations[depid]["in"].append((predrelation, govid))


        # Rule 5 :

        # Rule 6: advmod (without s_p) | amod ---> is-specialized-by
        # advmod (+ s_p ) --> p_c    e.g. : she ate her food happily
        for i, rels in enumerate(parse.dep):

            outr = [r[0] for r in rels["out"]]
            outindx = [r[1] for r in rels["out"]]

            objrels = {__DEPRELATIONS__["amod"], __DEPRELATIONS__["advmod"], __DEPRELATIONS__["nummod"]}

            for rel in objrels.intersection(set(outr)):

                govid = i
                depid = outindx[outr.index(rel)]

                # P_C is the predicate is a (relation) otherwise is-specialized-by
                if __RELATIONS__["s_p"] in [r[0] for r in relations[govid]["in"]]:
                    predrelation = __RELATIONS__["p_c"]
                else:
                    predrelation = __RELATIONS__["is-specialized-by"]

                # add subject --> predicate (is) relations
                relations[govid]["out"].append((predrelation, depid))
                relations[depid]["in"].append((predrelation, govid))



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










