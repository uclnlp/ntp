(('p0', 'X0', 'X1'), ('p1', 'X1', 'X0'))
0.947337	militaryactions(X,Y) :- duration(Y,X).
0.939915	duration(X,Y) :- militaryactions(Y,X).
0.886806	accusation(X,Y) :- duration(Y,X).
0.725322	violentactions(X,Y) :- duration(Y,X).
0.661134	blockpositionindex(X,Y) :- blockpositionindex(Y,X).
0.169211	duration(X,Y) :- duration(Y,X).
0.165436	duration(X,Y) :- duration(Y,X).
0.165032	duration(X,Y) :- duration(Y,X).
0.142308	aidenemy(X,Y) :- militaryactions(Y,X).
0.138	aidenemy(X,Y) :- militaryactions(Y,X).
0.136307	aidenemy(X,Y) :- militaryactions(Y,X).
0.13594	aidenemy(X,Y) :- duration(Y,X).
0.134607	aidenemy(X,Y) :- militaryactions(Y,X).
0.133663	aidenemy(X,Y) :- militaryactions(Y,X).
0.132718	aidenemy(X,Y) :- militaryactions(Y,X).
0.131585	aidenemy(X,Y) :- militaryactions(Y,X).
0.130054	aidenemy(X,Y) :- duration(Y,X).
0.128682	aidenemy(X,Y) :- militaryactions(Y,X).
0.124813	aidenemy(X,Y) :- militaryactions(Y,X).
0.124	aidenemy(X,Y) :- militaryactions(Y,X).

(('p0', 'X0', 'X1'), ('p1', 'X0', 'X2'), ('p2', 'X2', 'X1'))
0.381881	relemigrants(X,Y) :- lostterritory(X,Z), dependent(Z,Y).
0.332213	attackembassy(X,Y) :- dependent(X,Z), relemigrants(Z,Y).
0.217517	severdiplomatic(X,Y) :- duration(X,Z), duration(Z,Y).
0.213556	officialvisits(X,Y) :- aidenemy(X,Z), duration(Z,Y).
0.187817	severdiplomatic(X,Y) :- duration(X,Z), militaryactions(Z,Y).
0.12898	severdiplomatic(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.12869	severdiplomatic(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.125319	duration(X,Y) :- militaryactions(X,Z), aidenemy(Z,Y).
0.124278	severdiplomatic(X,Y) :- militaryactions(X,Z), duration(Z,Y).
0.123778	severdiplomatic(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.123119	severdiplomatic(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.122876	duration(X,Y) :- militaryactions(X,Z), relemigrants(Z,Y).
0.121096	duration(X,Y) :- militaryactions(X,Z), aidenemy(Z,Y).
0.120756	severdiplomatic(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.118933	severdiplomatic(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.118619	aidenemy(X,Y) :- militaryactions(X,Z), militaryactions(Z,Y).
0.118078	severdiplomatic(X,Y) :- militaryactions(X,Z), duration(Z,Y).
0.105824	duration(X,Y) :- aidenemy(X,Z), militaryactions(Z,Y).
0.104914	duration(X,Y) :- aidenemy(X,Z), militaryactions(Z,Y).
0.0772926	duration(X,Y) :- attackembassy(X,Z), militaryactions(Z,Y).

(('p0', 'X0', 'X1'), ('p1', 'X0', 'X1'))
0.937188	warning(X,Y) :- militaryactions(X,Y).
0.884529	dependent(X,Y) :- lostterritory(X,Y).
0.715856	relstudents(X,Y) :- students(X,Y).
0.552091	aidenemy(X,Y) :- warning(X,Y).
0.512543	pprotests(X,Y) :- boycottembargo(X,Y).
0.365957	students(X,Y) :- relemigrants(X,Y).
0.337928	aidenemy(X,Y) :- relemigrants(X,Y).
0.311478	lostterritory(X,Y) :- militaryactions(X,Y).
0.248008	expeldiplomats(X,Y) :- dependent(X,Y).
0.166303	duration(X,Y) :- duration(X,Y).
0.165555	duration(X,Y) :- duration(X,Y).
0.162268	duration(X,Y) :- duration(X,Y).
0.142869	lostterritory(X,Y) :- militaryactions(X,Y).
0.140903	aidenemy(X,Y) :- militaryactions(X,Y).
0.140711	lostterritory(X,Y) :- duration(X,Y).
0.139853	aidenemy(X,Y) :- militaryactions(X,Y).
0.13797	aidenemy(X,Y) :- militaryactions(X,Y).
0.134495	lostterritory(X,Y) :- militaryactions(X,Y).
0.131653	aidenemy(X,Y) :- militaryactions(X,Y).
0.109617	militaryalliance(X,Y) :- dependent(X,Y).

