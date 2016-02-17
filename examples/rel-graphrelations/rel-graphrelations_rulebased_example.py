__author__ = 'hadyelsahar'


from relationpreprocessor import *
sys.path.append('../../')

from sklearn.metrics import classification_report
from CNN import *
from bootstrapping.rulebased_relextractor import RuleBasedRelationExtractor

##########################################################################################
# Example for Utilization and Evaluation of the Rule based extractor of custom relations #
##########################################################################################

# p.S     :   dict {id:sentence}  the raw sentences
# p.X: array(dict) [{id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
# p.y     :   the correct labels                           - shape: nnx1
p = RelationPreprocessor()


X = [(p.S[e['sentence_id']], e["ent1"], e["ent2"]) for c, e in enumerate(p.X)]
y = p.y

y = np.array(y)

x_train = X[0:1700]
x_test = X[1700:]

y_train = y[0:1700]
y_test = y[1700:]


print "size of dataset is : %s" % len(y)
# print "now training..."
rb = RuleBasedRelationExtractor()

y_pred = rb.predict(x_test)

print classification_report(y_test, y_pred)


