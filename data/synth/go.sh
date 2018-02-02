seq 0 32 | awk '{ print "p(e" $1 ", e" $1 ")." }' > one.nl
seq 0 32 | awk '{ print "p(e" $1 ").\nq(e" $1 ")." }' > two.nl
seq 0 32 | awk '{ print "p(e" $1 ", e" $1 ").\nq(e" $1 ", e" $1 ").\nr(f" $1 ", f" $1 ")." }' > three.nl
