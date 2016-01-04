# Examples how to run vectorizers 

### cnn-mnist : 
- running the cnn solely and testing it's functionality over mnist handwritten digits image classification dataset.
- 
- How to run : `python mnist_example.py`
- you should get 98% as total f1 score and this evaluation report:
```
             precision    recall  f1-score   support

          0       0.98      0.99      0.98       980
          1       0.99      0.99      0.99      1135
          2       0.97      0.98      0.97      1032
          3       0.97      0.98      0.98      1010
          4       0.99      0.98      0.98       982
          5       0.96      0.99      0.97       892
          6       0.99      0.97      0.98       958
          7       0.97      0.97      0.97      1028
          8       0.98      0.96      0.97       974
          9       0.97      0.96      0.97      1009

avg / total       0.98      0.98      0.98     10000
```


### rel-semeval: 
- running the whole project (vectorizers + CNN) and testing it's functionality on semeval 2010 relation classification dataset.
- state of the art is 82.8% F-score by [Nguyen et al](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)


### rel-graphrelations:

