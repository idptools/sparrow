from sparrow.protein import Protein

P = Protein('MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP')

print('Demo 1')
print(P)
print(f"sparrow makes the most of Python's syntactic sugar e.g. we can use len() operator - e.g., len(P) will show the sequence length: {len(P)}")
print(P.predictor.disorder())
print(P.FCR)
