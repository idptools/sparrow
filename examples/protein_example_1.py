from sparrow.protein import Protein

P = Protein('MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP')

print('Demo 1')
print(P)
print("sparrow makes the most of Python's syntactic sugar e.g. we can use len() operator: %i"% len(P))
print(P.disorder)
print(P.is_IDP)
print(P.FCR)
