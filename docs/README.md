## Interpreting the Computation Graph

The Ts with the numbers informs you about the tensor dimensions

The first T50 in the goal means we have 50 predicate representations (each k dimensional)

The first rule (with T8544) tells us we are unifying the 50 goals with all 8544 facts in the KB

The result is a tensor with 8544x50 proof success scores

The next unification is more interesting… now we are unifying with 20 rules that have all the same structure

We get 20x50 unification success scores

Note the new variable bindings X1/T50… means we bound X1 to the 50 representations of the first arguments (entities) in the goal

Next, we need to prove the body of the rule T20(Y1, X1)

since we have 50 different success scores for each of the 20 body atoms, we need to construct 20*50=1000 goals -- this is where the tiling craziness begins.

Note the difference between inner and outer tiling to make sure we keep the correct reference to the unification success scores

Using the new goal (of 1000 atoms) we now unify with all facts in the KB — yielding 8544x1000 proof success scores

The next one is a slightly more complicated rule (a transitivity)

The mechanism is similar to the simple implication before… however now we are only keeping the 10max unifications after proving the first atom in the body of the rule

Note how under the TF Warning we instead of 8544x1000 keep only 10x1000 successes (and correspondingly only 10x1000 variable bindings for Z2)
