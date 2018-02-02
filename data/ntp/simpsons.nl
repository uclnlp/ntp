fatherOf(abe, homer).

parentOf(homer, lisa).

parentOf(homer, bart).

grandpaOf(abe, lisa).

grandfatherOf(abe, maggie).

grandparentOf(X, Y) :-
    grandfatherOf(X, Y).

parentOf(X, Y) :-
    fatherOf(X, Y).

grandfatherOf(X, Y) :-
    fatherOf(X, Z),
    parentOf(Z, Y).

grandchildOf(X, Y) :-
    grandparentOf(Y, X).