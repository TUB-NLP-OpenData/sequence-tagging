# active learning curves 
## uncertainty sampling vs. random sampling
* sequence-tagger: spacy-features + crfsuite
* 5 times 10 "steps"

### steps of 10% of trainset-size 
![0.1 steps](results/conll03_en_10percent/active_learning_curve.png)
### steps of 1% of trainset-size 
![0.01 steps](results/conll03_en_1percent/active_learning_curve.png)

## result
* entropy/uncertainty -based sampling seems not beneficial if model if dump (too few traindata or too shallow?)

