#!/usr/bin/env python
# coding: utf-8

# # Foreword

# This notebook contains functions used to forward simulate first-stage value functions using the method proposed by Bajari, Benkard and Levin (2007). The method uses conditional choice probabilities --- not estimated here --- added to random private shocks to simulate firm choice in a dynamic discrete choice.
# 
# We build a very simple dynamic model with only two observed variables to analyse movie theater behavior in response to a screen quota policy. Details concerning the model are beyond the scope of this exposition. Suffice to say $x_t$ corresponds to the state variable and each $t$ represents a movie session for a movie theater in the year 2018. The algorithm works the following way (for each multiplex):
# 1. At $t=1$, $x_1 = 0$. The algorithm gets week and day for $t=0$. With week information, it accesses all movies that were screened said week.
# 1. Having movies, day and $x_t$ information, we get kernel density estimates for each movie according to day/$x_t$ pair. Densities of all movies are summed up, such that probabilities are given by densities relative to total. In the Logit cases, relevant observation attributes are plugged in the model to get a probability prediction.
# 1. An extreme value error type I distribution is used to draw one shock for each movie.
# 1. Results for (2) and (3) are added together and the highest sum determines the "winner" movie
# 1. The expected occupation of the movie chosen in 4. is stored in an array
# 1. Private shock relative to the movie chosen in 4. is also stored in an array.
# 1. We record values for $\max(0,1 - x_t)$. When $t=0$, this equals $1$.
# 1. Finally, state transition is effected, according the state transition (known) function.
# 1. Repeat steps $1-9$ until we reach terminal state $t=T$.

# In[1]:


import numpy as np

# these are used to draw private shocks for each (movie) choice at each t and to introduce noise to estimates
from scipy.stats import gumbel_r, norm


# # Functions

# ## Deprecated

# ### Helper functions

# In[ ]:


# função para pegar um CPB (x), ano cine, semana cine, dia absoluto e estado e devolver uma densidade,
# a partir da densidade obtida por meio de uma kernel function calculada no arquivo principal (por isso globals)
# DEPRECATED em favor da função falt

def f_o1(x,ano,sem,dia,xt):
    # essa é só a maneira de acessar como a variável está definida para ano semana
    kde = globals()[f'{ano},{sem},{x}']
    # aqui, ela bota o np.exp "e^" porque a função score samples retorna o log da densidade
    return (np.exp(kde.score_samples([[dia, xt]])))[0]

# função para gerar as kdes, essencialmente ela percorre as semanas dos dados e obtem KDEs não paramétricas para cada
# obra que foi exibida naquela semana. o bandwidth foi escolhido depois de rodar algumas GridSearches com cross validation
# de máxima verosimilhança para ver qual bandwidth era mais adequado. eu peguei a moda dos resultados de uma amostra
# DEPRECATED

def get_kdes_o1(painel,painel_obras):
    from sklearn.neighbors import KernelDensity
    
    # isso é só uma lista com pares (ano, semana) porque 2017 e 2018 tem semana 52 e estão nos dados
    semanas = list(zip([2018 for x in range(52)],[x for x in range(1,53)]))
    semanas.insert(0, (2017,52))
    # dic para armazenar as funções de densidade
    d = {}
    
    for y,w in semanas:
        # pega painel da semana com id, dia e fração de cumprimento (xt_frac)
        painelzim = painel.query("ANO_CINEMATOGRAFICO==@y & SEMANA_CINEMATOGRAFICA==@w")[['cpb_id','DIA_abs','xt_frac']]
        for o in painel_obras.loc[(y,w),'cpb_id']:
            # para cada obra nessa semana, obtem uma kde no espaço dia vs. cumprimento proporcional
            ds = painelzim.query("cpb_id==@o")[['DIA_abs','xt_frac']].to_numpy()
            # armazenando as vars no dicionário
            d[f'{y},{w},{o}'] = KernelDensity(bandwidth=0.52).fit(ds)
    return d

# função para devolver uma array de densidades para uma array de CPBs
# DEPRECADA em favor da debaixo, que apenas pega o dicionário onde as densidades estão armazendas como argumento
# ver as funções f(.) e falt(.) e get_dens()

def get_dens_o1(xt, dia, sem, ano, obras):
    # só roda uma list comprehension passando a função element-wise em todas as obras em cartaz naquela semana
    dens = np.array([f(x, ano, sem, dia, xt) for x in obras])
    # determina a densidade relativa: densidade sobre soma das densidades, como probabilidade e pega o log da probabilidade
    return np.log(np.divide(dens, np.sum(dens)))


# ### Main functions

# #### Value functions

# In[ ]:


# função que calcula a média de #avg simulações de um complexo (#comp), usando um df #compobs com as observações de cada
# complexo, o painel de 2018 #painel e o painel segmentado com as obras de cada semana #painel_obras,
# DEPRECATED para c_val que toma o dicionário como argumento

def cval_o1(comp, avg, compobs, painel, painel_obras, obras):
    # pega a quantidade de observações, i.e., sessões do complexo analizado a partir de um dataframe compobs com essa info
    obs = compobs.loc[comp]
    #cria de antemão as arrays onde vamos armazenar os resultados de cada período simulado, nesse caso (sessoes x nº simulações)
    # sorteio total guarda o valor das gumbels para o filme escolhido
    sorteio_total = np.zeros((obs,avg))
    # ocup total pega o valor da ocupação da sala esperada para o filme selecionado
    ocup_total = np.zeros((obs,avg))
    # cota total pega a 1 - xt (fracionado) em cada período
    cota_total = np.ones((obs,avg))
    # separa o painel para o complexo com as infos necessárias
    # cump frac diz qual a fração da obrigação total do complexo será cumprida SE o filme escolhido for brasileiro naquela sess
    pc_np = painel[
            painel.REGISTRO_COMPLEXO == comp][['cump_frac','DIA_abs','SEMANA_CINEMATOGRAFICA','ANO_CINEMATOGRAFICO']].to_numpy()
    cf, da, sc, ac = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2]), vec_int(pc_np[:,3])
    # loopando e repetindo para a quantidade de simulações que queremos fazer para tirar a média (avg)
    for i in range(avg):
        # fração cumprida, começa sempre em 0
        xt = 0
        # ver acima para definição, mesma coisa, mas agora para 1 simulação
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        for t in range(obs):
            # pega as obras que passaram naquela semana
            cpb_array = painel_obras.loc[(ac[t],sc[t]),'cpb_id']
            # pega as log probabilidades condicionais de cada obra ser escolhida
            results = get_dens(xt, da[t], sc[t], ac[t], cpb_array)
            # sorteia as gumbels no tamanho da quantidade de obras daquela semana
            sorteio = gumbel_r.rvs(size=results.shape[0])
            # calcula o vencedor vendo o máximo 
            vencedor = np.argmax(results+sorteio)
            # registra o valor da gumbel do vencedor, que entra na função valor
            sort_venc[t] = sorteio[vencedor]
            # registra a ocupação esperada do vencedor e pega flag dizendo se ele é brasileiro
            ocup_venc[t], flag_br = obras.loc[(ac[t],sc[t],cpb_array[vencedor]),'flag']
            # registra cota do período
            cotas[t] = 1 - xt
            xt += flag_br*cf[t]
        # registra todos esses valores no ledger principal, para recomeçar o processo para a nova simulação
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    # já retorna a média de todas as simulações em uma array que tem (obs,3)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))


# #### Disturbed value functions

# In[ ]:


# função para calcular o valor "perturbado" do problema da firma, que inclui o "weighting factor" wf do cara
# essencialmente é exatamente a mesma função do cval, apenas com essa variação

def distval_o1(comp, avg, compobs, painel, painel_obras, wf, obras):
    obs = compobs.loc[comp]
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    pc_np = painel[
            painel.REGISTRO_COMPLEXO == comp][['cump_frac','DIA_abs','SEMANA_CINEMATOGRAFICA','ANO_CINEMATOGRAFICO']].to_numpy()
    cf, da, sc, ac = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2]), vec_int(pc_np[:,3])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        for t in range(obs):
            cpb_array = painel_obras.loc[(ac[t],sc[t]),'cpb_id']
            # essa é a única linha de diferença, ela transforma o array de CPBs em 1s e 0s, se brasileira ou não e soma esse 
            # viés para as obras brasileiras
            results = get_dens(xt, da[t], sc[t], ac[t], cpb_array) + vec_bras(cpb_array)*wf
            sorteio = gumbel_r.rvs(size=results.shape[0])
            vencedor = np.argmax(results+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_vencs[t], flag_br = obras.loc[(ac[t],sc[t],cpb_array[vencedor]),'flag']
            cotas[t] = 1 - xt
            xt += is_bras()*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))

# mesma coisa da versão anterior, com a diferença que ele pega o dicionário 'd' onde estão armazenadas as densidades!

def distval_o2(comp, avg, compobs, painel, painel_obras, wf, d, obras):
    obs = compobs.loc[comp]
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    pc_np = painel[
            painel.REGISTRO_COMPLEXO == comp][['cump_frac','DIA_abs','SEMANA_CINEMATOGRAFICA','ANO_CINEMATOGRAFICO']].to_numpy()
    cf, da, sc, ac = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2]), vec_int(pc_np[:,3])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        for t in range(obs):
            cpb_array = painel_obras.loc[(ac[t],sc[t]),'cpb_id']
            results = get_densalt(xt, da[t], sc[t], ac[t], cpb_array, d) + vec_bras(cpb_array)*wf
            sorteio = gumbel_r.rvs(size=results.shape[0])
            vencedor = np.argmax(results+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t], flag_br = obras.loc[(ac[t],sc[t],cpb_array[vencedor]),'flag']
            cotas[t] = 1 - xt
            xt += flag_br*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))


# ## Stable

# ### Helper functions

# #### Density/probability from dict

# In[ ]:


# this functions take info about movies, dates and states and return Kernel Density estimates for each. note that they get run
# inside the get_dens function defined below
    # args:
    # x = movie, given by id
    # sem = cinematographic week
    # dia = day
    # xt = quota fulfilled up to time t
    # d = dictionary storing KDEs

def f(x,sem,dia,xt,d):
    kde = d[f'{sem},{x}'] # getting KDE estimates for week/movie pair
    return (np.exp(kde.score_samples([[dia, xt]])))[0] # scoring it according to position in day/quota fulfillment space

def f_noise(x,sem,dia,xt,d): # same as above, but includes random normal error inside estimates
    kde = d[f'{sem},{x}']
    return (np.exp(kde.score_samples([[dia, xt]])))[0]*(1+norm.rvs(1))


# #### Get functions

# In[ ]:


# this function creates Kernel Density Estimates used in the value functions below and return a dictionary used as argument
# bandwidth estimates are obtained through GridSearchCV and cross validation methods
# KDEs are calculated for each movie/week pair
# painel = 2018 panel with all session-level observations
# np_obras = see np_movies in cval below

def get_kdes(painel,np_obras):
    from sklearn.neighbors import KernelDensity
    
    d = {} # dict to store results
    cpb_index = np.arange(np_obras.shape[0]) # check cpb_index in the cval function
    
    # looping over all weeks
    for w in range(53):
        # filter panel with all 2018 obs for week 'w' and only takes 'cpb_id', day and fractional fulfillment of quotas
        # see pc_np argument for cval below
        painelzim = painel.query("SEMANA_CINEMATOGRAFICA==@w")[['cpb_id','DIA_abs','xt_frac']]
        for o in cpb_index[np.where(np_obras[:,w] > 0, True, False)]:
            # now more narrowly defined for each movie id
            ds = painelzim.query("cpb_id==@o")[['DIA_abs','xt_frac']].to_numpy()
            # computing and storing KDEs
            d[f'{w},{int(o)}'] = KernelDensity(bandwidth=0.52).fit(ds)
    return d


# In[ ]:


# gets density using several variables and density using funcion f above, then calculates (log probability) as relative 
# density to the sum of all other movies' densities
# xt = fractional fulfillment of screen quota obligations up to time t
# week = cinematographic week
# day = absolute day
# movies = array saying which movies were on screen
# d = dictionary with KDE results, see d above

def get_dens(xt, week, day, movies, d):
    dens = np.array([f(x, week, day, xt, d) for x in movies]) # see function f
    return np.log(np.divide(dens, np.sum(dens))) # get log of each relative density (i.e. density / sum of densities)

# this is exactly as above, but using noisy (random standard normal) errors in the density estimates
# see distval_noise below

def get_dens_noise(xt, sem, dia, obras, d):
    dens = np.array([f_noise(x, sem, dia, xt, d) for x in obras])
    return np.log(np.divide(dens, np.sum(dens)))


# In[ ]:


# gets distval bias for a given movie theather id
# args:
# c = movie theater id
# compobs = dataframe with ids vs. number of sessions (t)
# painel = panel to get pc_np (see main functions)
# wfs = weighting factors (see 2.2.2.2)
# np_obras = array containing movies per week and avg. seat occupation (see main functions)

def get_distval(c, compobs, avg, painel, np_obras, wfs, njobs):
    from joblib import Parallel, delayed
    pc_np = pc_np = painel[painel.REGISTRO_COMPLEXO == c][['cump_frac','DIA_abs','SEMANA_CINEMATOGRAFICA']].to_numpy()
    
    results = np.zeros((compobs.loc[c],3,len(wfs))) # creating array to store results
    
    results[:,:,0], results[:,:,1], results[:,:,2], results[:,:,3] = Parallel(n_jobs = njobs, backend='multiprocessing')(
        delayed(distval)(c, avg, compobs.loc[c], pc_np, np_obras, i, d) for i in wfs) # processing
    
    for n in range(len(wfs)): # applying discount factors
        for i in range(3):
            results[:,i,n] = np.multiply(results[:,i,n],painel.loc[(painel.REGISTRO_COMPLEXO == 6586), 'beta'].values)
    
    return results


# #### Vectorized

# In[ ]:


# this function only indicates whether a id corresponds to a Brazilian or foreign movies

def is_bras(x):
    if x < 356: # unique Ancine ids start with B's for Brazilian movies and E's for foreign ones (from "Estrangeiro")
                # meaning that when we order them to get simple number ids B's come first, that's why 356 is the threshold
        return 1
    else:
        return 0

def vec_bras(x): # same thing but applying is bras function to a whole array at once for efficiency reasons
    vec = np.vectorize(is_bras)
    return vec(x)

def vec_int(x): # simple vectorizing of int function
    vec = np.vectorize(int)
    return vec(x)

def vec_zero(x): # vectorizing turning negative ints into zero, used for transforming negative remaining quotas into 0
                 # for example if a movie theater screened 150% of its obligations 1 - x_t = - 0.5
    vec = np.vectorize(lambda x: 0 if x < 0 else x)
    return vec(x)


# #### Misc

# In[ ]:


# this function is used to transform yearly interest rates 'r' into daily discount factors to discount the value functions

def daily_interest(r, days=365):
    return (1+r/100)**(1/float(days)) - 1


# ### Main functions

# #### Value function and counterfactual

# In[ ]:


# This is the main value function used, and it takes several arguments:
# comp = movie theather complex id
# avg = number of computations to average out, this is a requirement of BBL method to converge to the true value
# obs = no. of movie sessions for complex 'comp' in year 2018, retrieved from data
# pc_np = numpy array with information for each 'obs' informing (absolute) day of year, (cinematographic) week, 
    # and state transition
# np_movies = array containing movies screened each week and info with average occupation for each one
# d = dictionary containing Kernel Density Estimates for each movie in the time, x_t space for each week, used to
    # compute first stage Conditional Choice Probabilities

def cval(comp, avg, obs, pc_np, np_movies, d):
    # first we create arrays to store private shocks, average occupation for the chosen movie and quota fulfilled for each
    # period t, and for every simulation done as determined by the number to average out
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    # as stated above, we store each column of pc_np
    # cf = fractional fulfillment of quotas, should the chosen movie in session t be Brazilian
    # da = absolute day
    # sc = cinematographic week
    cf, da, sc = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2])
    # movie ids for all movies have been transformed from a string to a number from 0 to 908, see below
    cpb_index = np.arange(np_movies.shape[0])
    # looping over the process for the number of times chosen as 'avg' argument
    for i in range(avg):
        # quota fulfillment starts at 0%
        xt = 0
        # here we create arrays to store results same as above, but for the present simulation
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        # to see movie ids displayed each week, we start with week 0 of 'np_movies[:,0]' and look only for the rows
        # where values are not 0. in other words, np_movies stores in each columns information regarding all movies screened
        # noting 0 if the movie was not on screens that week and the expected occupation value otherwise. rows are indexed
        # from 0 to 908. 'cpb_index' is constructed to rebuild this index in the np.array object
        cpb_array = cpb_index[np.where(np_movies[:,0] > 0, True, False)]
        for t in range(obs):
            # this only changes the movies when the week changes as we go through the t sessions
            if t > 0:
                if sc[t] > sc[t-1]:
                    cpb_array = cpb_index[np.where(np_movies[:,sc[t]] > 0, True, False)]
            # results computes log probabilities for each movie on screen, as recorded by 'cpb_array'
            # for details, see helper function get_dens in the helper functions
            # arguments include state, week, day, besides movie and the stored dictionary of first stage estimators
            results = get_dens(xt, sc[t], da[t], cpb_array, d)
            # random shocks are drawn from extreme type I value distribution for each movie (thus the shape)
            sorteio = gumbel_r.rvs(size=results.shape[0])
            # chosen movie is determined by the max of results and random shocks
            vencedor = np.argmax(results+sorteio)
            # now we store occupation, shock and screen quota information with the chosen movie
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t] = np_movies[cpb_array[vencedor],sc[t]]
            # see that the chosen movie does not affect quota fulfillment in state t, only in t+1
            cotas[t] = 1 - xt
            # function is_bras checks if the movie is Brazilian according to its unique id, returning 1 if True
            # to get state transition we only look for fractional fulfillment previously calculated
            xt += is_bras(cpb_array[vencedor])*cf[t]
        # now we record full results of 1 round of simulation in the initial array
        # note that vec_zero now transforms all negative remaining quota values to 0
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    # return the average of each column
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))


# In[ ]:


# This is essentially the same as above, but now we use logit first stage CCP estimators, which requires some changes
# Only changes are noted. For omitted details, check cval above.
# Different args are:
# regs = dictionary with fitted logit models for each week (using sklearn.LogisticRegression)
# cols = columns index for logit explanatory variables (models have interacted values and many fixed-effects so we need an
    # index to get rid of dataframes that require more memory usage)

def cval_logit(comp, avg, obs, pc_np, np_obras, regs, cols):
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    cf, da, sc = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        w_reg = regs['reg_0'] # choosing week 0 fitted model from dict
        w_col = cols['semana_0'].values # same for column index
        cpb_array = w_reg.classes_ # movies each week are dependent variables of multinomial logit stored under classes
        # attribute of sklearn.LogisticRegression.fit()
        for t in range(obs):
            # same mechanism as before, rewriting variables when week changes
            if t > 0:
                if sc[t] > sc[t-1]:
                    w_reg = regs[f'reg_{sc[t]}']
                    w_col = cols[f'semana_{sc[t]}'].values
                    cpb_array = w_reg.classes_
            # here we prepare the array of explanatory variables to feed the logit model, using day, movie theater id and frac
            # fulfillment
            log_proba = np.select(
            [w_col == f'DIA_abs_{da[t]}', w_col == f'REGISTRO_COMPLEXO_{comp}', w_col == f'REGISTRO_COMPLEXO_{comp}:xt_frac', w_col == f'DIA_abs_{da[t]}:xt_frac'],
                [1,1,xt,xt])
            # reshaping for the expected shape of sklearn
            log_proba = log_proba.reshape(1,log_proba.shape[0])
            log_proba[0,0] = xt
            # same as cval, now we use predict log proba method directly from sklearn for all movies
            results = w_reg.predict_log_proba(log_proba).flatten()
            sorteio = gumbel_r.rvs(size=results.shape)
            vencedor = np.argmax(results+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t] = np_obras[cpb_array[vencedor],sc[t]]
            cotas[t] = 1 - xt
            xt += is_bras(cpb_array[vencedor])*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))


# In[ ]:


# conterfactual functions do the same as the above functions, but now they simulate paths using parameter estimates
# obtained from second stage estimation using the BBL algorithm

def counterfactual(comp, avg, pc_np, np_obras, theta_1):
    xts = np.zeros((avg,))
    cf, sc = pc_np[:,0], vec_int(pc_np[:,2])
    cpb_index = np.arange(np_obras.shape[0])
    for i in range(avg):
        xt = 0
        for s in range(53):
            ocup_obras = np_obras[:,s][np.where(np_obras[:,s] > 0, True, False)]
            cpb_array = cpb_index[np.where(np_obras[:,s] > 0, True, False)]
            obs_semana = np.sum(np.where(sc == s, True, False))
            sorteio = gumbel_r.rvs(size=(obs_semana, ocup_obras.shape[0]))
            
            ocs = np.zeros(sorteio.shape)
            ocs[:,:] = np.multiply(theta_1, ocup_obras)
            
            resultados = np.sum([ocs, sorteio], axis=0)
            idx_vencedores = np.argmax(resultados, axis=1)
            vencedores = cpb_array[idx_vencedores]
            
            xt += np.sum(np.multiply(vec_bras(vencedores), cf[:obs_semana]))
        xts[i] = xt
    return xts.mean()


# #### Disturbed value functions

# In[ ]:


# Bajari, Benkard and Levin required disturbed value functions to obtain parameter estimates in the second stage
# the idea is that true parameters will minimize equilibrium violations, i.e., when disturbed value functions
# yield higher equilibrium values than the true ones
# this function only takes in a new argument:
# wf = weighting factor, used to systematically bias log probs of brazilian movies upwards or downwards
# for explanations, check cval function

def distval(comp, avg, obs, pc_np, np_obras, wf, d):
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    cf, da, sc = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2])
    cpb_index = np.arange(np_obras.shape[0])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        cpb_array = cpb_index[np.where(np_obras[:,0] > 0, True, False)]
        for t in range(obs):
            if t > 0:
                if sc[t] > sc[t-1]:
                    cpb_array = cpb_index[np.where(np_obras[:,sc[t]] > 0, True, False)]
            # this is the only difference, brazilian movies picked by vec_bras as 1 are multiplied to bias log probs
            results = get_dens(xt, sc[t], da[t], cpb_array, d) + vec_bras(cpb_array)*wf
            sorteio = gumbel_r.rvs(size=results.shape[0])
            vencedor = np.argmax(results+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t] = np_obras[cpb_array[vencedor],sc[t]]
            cotas[t] = 1 - xt
            xt += is_bras(cpb_array[vencedor])*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))

# same idea as the function before, but altering for logit first-stage estimators

def distval_logit(comp, avg, obs, pc_np, np_obras, wf, regs, cols):
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    cf, da, sc = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        w_reg = regs['reg_0']
        w_col = cols['semana_0'].values
        cpb_array = w_reg.classes_
        for t in range(obs):
            if t > 0:
                if sc[t] > sc[t-1]:
                    w_reg = regs[f'reg_{sc[t]}']
                    w_col = cols[f'semana_{sc[t]}'].values
                    cpb_array = w_reg.classes_
            log_proba = np.select(
            [w_col == f'DIA_abs_{da[t]}', w_col == f'REGISTRO_COMPLEXO_{comp}', w_col == f'REGISTRO_COMPLEXO_{comp}:xt_frac', w_col == f'DIA_abs_{da[t]}:xt_frac'],
                [1,1,xt,xt])
            log_proba = log_proba.reshape(1,log_proba.shape[0])
            log_proba[0,0] = xt
            results = w_reg.predict_log_proba(log_proba).flatten() + vec_bras(cpb_array)*wf
            sorteio = gumbel_r.rvs(size=results.shape)
            vencedor = np.argmax(results+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t] = np_obras[cpb_array[vencedor],sc[t]]
            cotas[t] = 1 - xt
            xt += is_bras(cpb_array[vencedor])*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))


# In[1]:


# this is a second alternative to produce disturbed value functions. in this case que just add random noise to 
# log prob estimates. essentially the functions multiplies each log prob estimate by 1+ standard random noise

def distval_noise(comp, avg, obs, pc_np, np_obras, d):
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    cf, da, sc = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2])
    cpb_index = np.arange(np_obras.shape[0])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        cpb_array = cpb_index[np.where(np_obras[:,0] > 0, True, False)]
        for t in range(obs):
            if t > 0:
                if sc[t] > sc[t-1]:
                    cpb_array = cpb_index[np.where(np_obras[:,sc[t]] > 0, True, False)]
            # the only change is mediated by this different get_density function that introduces normal errors
            results = get_dens_noise(xt, sc[t], da[t], cpb_array, d)
            sorteio = gumbel_r.rvs(size=results.shape[0])
            vencedor = np.argmax(results+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t] = np_obras[cpb_array[vencedor],sc[t]]
            cotas[t] = 1 - xt
            xt += is_bras(cpb_array[vencedor])*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))

# exactly the same as before but for logit CCPs

def distval_noise_logit(comp, avg, obs, pc_np, np_obras, regs, cols):
    sorteio_total = np.zeros((obs,avg))
    ocup_total = np.zeros((obs,avg))
    cota_total = np.ones((obs,avg))
    cf, da, sc = pc_np[:,0], vec_int(pc_np[:,1]), vec_int(pc_np[:,2])
    for i in range(avg):
        xt = 0
        sort_venc = np.zeros((obs,))
        ocup_venc = np.zeros((obs,))
        cotas = np.ones((obs,))
        w_reg = regs['reg_0']
        w_col = cols['semana_0'].values
        cpb_array = w_reg.classes_
        for t in range(obs):
            if t > 0:
                if sc[t] > sc[t-1]:
                    w_reg = regs[f'reg_{sc[t]}']
                    w_col = cols[f'semana_{sc[t]}'].values
                    cpb_array = w_reg.classes_
            log_proba = np.select(
            [w_col == f'DIA_abs_{da[t]}', w_col == f'REGISTRO_COMPLEXO_{comp}', w_col == f'REGISTRO_COMPLEXO_{comp}:xt_frac', w_col == f'DIA_abs_{da[t]}:xt_frac'],
                [1,1,xt,xt])
            log_proba = log_proba.reshape(1,log_proba.shape[0])
            log_proba[0,0] = xt        
            results = w_reg.predict_log_proba(log_proba).flatten()
            # once again, this is the only difference
            noise = np.multiply(results, 1+norm.rvs(size=results.shape))
            sorteio = gumbel_r.rvs(size=results.shape)
            vencedor = np.argmax(noise+sorteio)
            sort_venc[t] = sorteio[vencedor]
            ocup_venc[t] = np_obras[cpb_array[vencedor],sc[t]]
            cotas[t] = 1 - xt
            xt += is_bras(cpb_array[vencedor])*cf[t]
        sorteio_total[:,i], ocup_total[:,i], cota_total[:,i] = sort_venc, ocup_venc, vec_zero(cotas)
    return np.column_stack((sorteio_total.mean(axis=1), ocup_total.mean(axis=1), cota_total.mean(axis=1)))


# # Export to script

# In[1]:


#!jupyter nbconvert --to script bbl.ipynb

