(('p0', 'X0', 'X1'), ('p1', 'X0', 'X1'))
0.942011	co-occurs_with(X,Y) :- degree_of(X,Y).
0.862528	prevents(X,Y) :- treats(X,Y).
0.82948	measures(X,Y) :- analyzes(X,Y).
0.477553	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.476172	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.474864	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.474651	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.472924	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.471849	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.468755	developmental_form_of(X,Y) :- developmental_form_of(X,Y).
0.465784	adjacent_to(X,Y) :- adjacent_to(X,Y).
0.465047	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.464889	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.464707	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.464532	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.463158	developmental_form_of(X,Y) :- adjacent_to(X,Y).
0.442301	isa(X,Y) :- co-occurs_with(X,Y).
0.420866	method_of(X,Y) :- method_of(X,Y).
0.406279	analyzes(X,Y) :- measures(X,Y).
0.402711	developmental_form_of(X,Y) :- method_of(X,Y).

(('p0', 'X0', 'X1'), ('p1', 'X0', 'X2'), ('p1', 'X2', 'X1'))
0.990477	interacts_with(X,Y) :- interacts_with(X,Z), interacts_with(Z,Y).
0.764667	developmental_form_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.586205	isa(X,Y) :- conceptual_part_of(X,Z), conceptual_part_of(Z,Y).
0.428585	consists_of(X,Y) :- derivative_of(X,Z), derivative_of(Z,Y).
0.398652	conceptually_related_to(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.395944	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.388299	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.383544	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.380323	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.376661	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.374857	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.373426	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.372877	method_of(X,Y) :- developmental_form_of(X,Z), developmental_form_of(Z,Y).
0.367539	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.347574	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.345293	conceptually_related_to(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.335167	treats(X,Y) :- manages(X,Z), manages(Z,Y).
0.313067	method_of(X,Y) :- conceptually_related_to(X,Z), conceptually_related_to(Z,Y).
0.171882	exhibits(X,Y) :- practices(X,Z), practices(Z,Y).
0.167955	conceptually_related_to(X,Y) :- precedes(X,Z), precedes(Z,Y).

(('p0', 'X0', 'X1'), ('p1', 'X0', 'X2'), ('p2', 'X2', 'X1'))
0.981781	performs(X,Y) :- interacts_with(X,Z), performs(Z,Y).
0.98061	part_of(X,Y) :- part_of(X,Z), interacts_with(Z,Y).
0.942	isa(X,Y) :- isa(X,Z), isa(Z,Y).
0.923871	method_of(X,Y) :- isa(X,Z), method_of(Z,Y).
0.875344	result_of(X,Y) :- isa(X,Z), result_of(Z,Y).
0.87499	associated_with(X,Y) :- associated_with(X,Z), degree_of(Z,Y).
0.677233	conceptual_part_of(X,Y) :- adjacent_to(X,Z), conceptual_part_of(Z,Y).
0.653239	location_of(X,Y) :- developmental_form_of(X,Z), location_of(Z,Y).
0.344066	conceptual_part_of(X,Y) :- adjacent_to(X,Z), adjacent_to(Z,Y).
0.342012	evaluation_of(X,Y) :- evaluation_of(X,Z), method_of(Z,Y).
0.330638	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.330059	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.318259	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.312912	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.308828	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.293754	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.247132	measures(X,Y) :- measures(X,Z), conceptual_part_of(Z,Y).
0.246786	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.246704	conceptually_related_to(X,Y) :- connected_to(X,Z), adjacent_to(Z,Y).
0.211109	measures(X,Y) :- measures(X,Z), degree_of(Z,Y).

