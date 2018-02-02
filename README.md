# End-to-End Differentiable Proving
This is an implementation of the paper [End-to-End Differentiable Proving](http://papers.nips.cc/paper/6969-end-to-end-differentiable-proving.pdf). For a high-level introduction, see the [NIPS oral](https://www.youtube.com/watch?v=WWWQXTb_69c&t=1700s), [slides](https://rockt.github.io/pdf/rocktaschel2017end-slides.pdf) and [poster](https://rockt.github.io/pdf/rocktaschel2017end-poster.pdf).


## Disclaimer
Please note that this software is not maintained. It is highly-experimental research code, not well documented and we provide no warranty of any kind. Use at your own risk!

## Data Format

Data for the NTP is in `nl` format - basically Prolog syntax:

```shell
ntp$ head data/countries/countries.nl
locatedIn(palau,micronesia).
locatedIn(palau,oceania).
locatedIn(maldives,southern_asia).
locatedIn(maldives,asia).
locatedIn(brunei,south-eastern_asia).
locatedIn(brunei,asia).
neighborOf(brunei,malaysia).
locatedIn(japan,eastern_asia).
locatedIn(japan,asia).
locatedIn(netherlands,western_europe).
```

- `*.nl` files represent *facts and rules* (example of a rule: `isa(X,Y) :- isa(X,Z), isa(Z,Y)`)

- `*.nlt` files represent *rule templates* (example of a rule template: `#1(X,Y) :- #2(X,Z), #3(Z,Y)`)

```shell
ntp$ cat data/ntp/simpsons.nlt
5   #1(X, Y) :- #2(X, Y).

5   #1(X, Y) :- #1(Y, X).

5   #1(X, Y) :-
    #2(X, Z),
    #2(Z, Y).
```

## Running

The main file for running NTP is `ntp/experiments/learn.py` which takes the path to a configuration file as argument.

## Code Structure

The core implementation of the NTP can be found [here](https://github.com/uclmr/ntp/blob/master/ntp/prover.py).

The base models (neural link predictors) are implemented [here](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L253).

Imortant "modules" are [unify](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L195),
[this one](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L195)
and [this one](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L470).
It should pretty much reflect the pseudocode in the paper.

The tricky part is the tiling of batched representations for batch proving -
check out [this](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L160).

However, this *tiling* needs to happen at various points in the code,
e.g. [here](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L492)

Implementation of tiling (and multiplexing)
[here](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L319) and
[here](https://github.com/uclmr/ntp/blob/master/ntp/prover.py#L346).

An important trick in NTP for proving in larger KBs and usin complex rules, is the Kmax heuristic,
implemented [here](https://github.com/uclmr/ntp/blob/master/ntp/kmax.py).

There is a *symbolic prover implementation* [here](https://github.com/uclmr/ntp/blob/master/ntp/tp.py)
- it is probably worthwile to look at it first, and compare to NTP.

## Test

```shell
nosetests
```

## Contributors
- [Tim Rocktäschel](https://rockt.github.com)
- [Sebastian Riedel](http://www.riedelcastro.org/)
- [Pasquale Minervini](http://www.neuralnoise.com/)
- [Johannes Welbl](https://jowel.gitlab.io/welbl/)

## Citation
```
@inproceedings{rocktaschel2017end,
  author    = {Tim Rockt{\"{a}}schel and
               Sebastian Riedel},
  title     = {End-to-end Differentiable Proving},
  booktitle = {Advances in Neural Information Processing Systems 30: Annual Conference
               on Neural Information Processing Systems 2017, 4-9 December 2017,
               Long Beach, CA, {USA}},
  pages     = {3791--3803},
  year      = {2017},
  url       = {http://papers.nips.cc/paper/6969-end-to-end-differentiable-proving},
}
```
