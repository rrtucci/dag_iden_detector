# DAG Iden Detector

Iden= Identifiability

## Code History
This repo is a superset of a previous github repo of mine:

https://github.com/rrtucci/napkin-do-calc

The main difference between this repo and the previous one
is the file **identification.py** and 
all the new jupyter notebooks and pics.
(see **jupyter_notebooks** folder)

The code pertaining "potentials"
in this repo and the previous one comes from my previous 
repo

https://github.com/artiste-qb-net/quantum-fog


## Do Calculus Background
Judea Pearl has defined 
the concept of "identifiability" for
a do-query $P(y|do(\underline{x})=x)$
with respect to a Bayesian Network that I like to 
call the Original Promise (OP). Pearl has
also given 3 rules of "Do Calculus"
which can be applied to 
the OP to decide whether a
do-query for the OP is identifiable or not.
If it is identifiable, the 3 Rules
also yield a so called "adjustment formula".

An adjustment formula is
a formula that equals $P(y|do(\underline{x})=x)$
but 

* it contains no do operators 
* it contains no probabilities
of hidden variables, even when the OP has some. 
* It can be calculated from DATA
OBTAINED WITHOUT CONDUCTING AN RCT.

## What this code can and cannot do
This repo contains python
code for checking numerically (on 
an OP with random CPTs)
whether an adjustment formula is 
correct or not. The code cannot decide
whether a do-query is identifiable or not,
nor can it derive new adjustment formulae.
What it can do is to check whether an
adjustment formula is correct or not.

## Our Findings

Pearl has given adjustment formulae (AF) that
he calls the backdoor AF and frontdoor AF.
Cinelli has given an AF for the "Napkin" OP [here](https://stats.stackexchange.com/questions/514615/do-calculus-for-causal-diagram-7-5-from-the-book-of-why-napkin-problem) 


Our findings are presented in jupyter notebooks in the
**jupyter_notebooks** folder. See the 
[SUMMARY notebook](https://github.com/rrtucci/dag_iden_detector/blob/master/jupyter_notebooks/SUMMARY.ipynb)
for a summary of each notebook

We show that:
* the backdoor AF is correct
* the frontdoor AF is correct
* Cinelli's AF for the napkin OP is INCORRECT
* We guess an AF for the napkin OP and show 
that it's correct




