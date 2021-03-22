# BBL
Deployment of Bajari, Benkard &amp; Levin (2007) estimation algorithm

This project contains functions used to forward simulate first-stage value functions using the method proposed by Bajari, Benkard and Levin (2007). The method uses conditional choice probabilities --- not estimated here --- added to random private shocks to simulate firm choice in a dynamic discrete choice.

We build a very simple dynamic model with only two observed variables to analyse movie theater behavior in response to a screen quota policy. Details concerning the model are beyond the scope of this exposition. Suffice to say $x_t$ corresponds to the state variable and each $t$ represents a movie session for a movie theater in the year 2018. The algorithm works the following way (for each multiplex):
1. At $t=1$, $x_1 = 0$. The algorithm gets week and day for $t=0$. With week information, it accesses all movies that were screened said week.
1. Having movies, day and $x_t$ information, we get kernel density estimates for each movie according to day/$x_t$ pair. Densities of all movies are summed up, such that probabilities are given by densities relative to total. In the Logit cases, relevant observation attributes are plugged in the model to get a probability prediction.
1. An extreme value error type I distribution is used to draw one shock for each movie.
1. Results for (2) and (3) are added together and the highest sum determines the "winner" movie
1. The expected occupation of the movie chosen in 4. is stored in an array
1. Private shock relative to the movie chosen in 4. is also stored in an array.
1. We record values for $\max(0,1 - x_t)$. When $t=0$, this equals $1$.
1. Finally, state transition is effected, according the state transition (known) function.
1. Repeat steps $1-9$ until we reach terminal state $t=T$.
