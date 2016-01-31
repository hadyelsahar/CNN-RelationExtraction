__author__ = 'hadyelsahar'

from sklearn.base import BaseEstimator, ClassifierMixin
import os
from corenlpclient import *

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
    "nsubj": "nsubj"
}

class RuleBasedRelationExtractor(BaseEstimator, ClassifierMixin):

    def __init__(self, rawdir="./rawfiles", outputdir="./output"):

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

        # for i in os.listdir(self.rawdir):
            i = self.rawdir+"/sample.txt"
            s = open(i).read()

            # Getting dependency parse tree, NER and COREF
            parse = self.parser.annotate(s)

            # Generate new relations for the custom representations
            relations = self.extractrelations(s, parse)

            # Export the custom relations in a .ann file and write both .txt and .ann into the outdir
            # self.save_in_brat_format(i.replace(".txt", ""), s, relations)

            return relations


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




        #############################################################
        # Removal of redundant relations and relations within phrases
        #############################################################

        relations = [{"in": list(set(i["in"])), "out":list(set(i["out"]))} for i in relations]

        return relations









    def save_in_brat_format(self,fname, s, relations):

        pass







