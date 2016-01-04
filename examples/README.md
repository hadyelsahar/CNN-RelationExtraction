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
- running the whole project (vectorizers + CNN) and testing it's functionality on [semeval 2010 relation classification dataset](http://delivery.acm.org/10.1145/1630000/1621986/p94-hendrickx.pdf?ip=79.94.243.43&id=1621986&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=706255923&CFTOKEN=18070285&__acm__=1451906539_3895018be4fb7c5e7a98afdb2834716d).
- semeval 2010 Rel classification [dataset download](http://semeval2.fbk.eu/semeval2.php?location=data) : 
- state of the art is 82.8% F-score by [Nguyen et al](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)



### rel-graphrelations:

