#!/usr/bin/python
import numpy as np
import os, sys
import matplotlib
#mpl.use('Agg')

matplotlib.use('Agg')


import matplotlib.pyplot as plt
from pylab import *
from threading import Thread
from subprocess import PIPE, Popen
import scipy.optimize as op
import emcee
import signal
import corner
import dynesty_1_2 as dynesty
import time
import timeit
import dill


'''
Trifon Trifonov 2020
'''

THIRD = 1.0/3.0
PI    = 3.14159265358979e0
TWOPI = 2.0*PI
GMSUN = 1.32712497e20
AU=1.49597892e11
G  = 6.67384e-11 

start = timeit.default_timer()
########################### For nice plotting ##################################

mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
mpl.rcParams['xtick.major.pad']='1'
mpl.rcParams['ytick.major.pad']='2'


# set tick width
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 2

mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 2


rc('text',usetex=True)
font = {'family' : 'normal','weight' : 'black','size': 22,'serif':['Helvetica']}
rc('font', **font)



 

##################### innitial guess of parametrs to vary ######################

 
best_fit_est   = np.array([0.573, 1.57, 1050.74, 0.2366, 5.04, 13.2, 2700.0]) # M1, M2, Per[days], ecc, R2, L2
e_best_fit_est = np.array([0.015, 0.06, 0.20, 0.0003, 0.10, 0.30, 350.0 ] )
 
#par    = np.array([2.3, 1.4, 2500.0, 0.8, 10000.0]) # init_M1, init_M2, init_P, init_e, init_Bw (used for optimisation)
par    = np.array([2.4, 1.4, 750.0, 0.8, 16000.0]) # init_M1, init_M2, init_P, init_e, init_Bw (used for optimisation)
 
flag   = np.array([1, 1, 1, 1, 1]) # flags; 1 - fit, 0 - fix.

#bounds = np.array([[1.7,3.0], [0.4,1.55], [365.0,40000], [0.70,0.999], [0.0,30000.0]]) # prior range adopted for fixed Bw (nu Oct)
bounds = np.array([[1.7,3.0], [0.4,1.55], [0.0,1500], [0.0,0.999], [3000.0,40000.0]]) # prior range adopted for free Bw (nu Oct)
 
#------------------------- Parameter names --------------------#
el_str = np.array([r'$M_{\rm A}$ [M$_\odot$]', r'$M_{\rm B}$ [M$_\odot$]', r'$P_{\rm binary}$ [days]',r'$e_{\rm binary}$', r'$B_{\rm w}$'])


mod = "nest" # for nested sampling


########################## run the wrapper #####################################



class FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.
    """
    def __init__(self, f, args, kwargs=None):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        try:
            result = self.f(x, *self.args, **self.kwargs)
            #print(x, result)
            return result
        except:  # pragma: no cover
            import traceback
            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def run_command_with_timeout(args, secs):
    proc = Popen(args, shell=True, preexec_fn=os.setsid, stdout=PIPE)
    proc_thread = Thread(target=proc.wait)
    proc_thread.start()
    proc_thread.join(secs)
    text = proc.communicate()[0]
    flag = 1
    if proc_thread.is_alive():
        print(proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:

            print('Process #{} killed after {} seconds'.format(proc.pid, secs))
            flag = -1
            #text = '0 0 0 0'
            return text.decode('utf-8'),flag
    #return proc, flag , text.decode('utf-8')
    return text.decode('utf-8'),flag

################  concentrate parameter in arrays ################# 
def concentrate_par(par,flag,bounds,el_str):
     
    f = [idx for idx in range(len(flag)) if flag[idx] ==1 ]

    p = []  #'p" are the fitted parameters
    b = []
    e = []

    for j in range(len(par)):
        if flag[j] > 0:
            p.append(par[j])
            b.append(bounds[j])
            e.append(el_str[j])

    return p,f,b,e 


 
 
 
################################ wrapper #######################################   

def BSE(p, par,flag_ind, best_fit_est, e_best_fit_est):


    
    for j in range(len(p)):
        par[flag_ind[j]] = p[j] 

    rand_file_name = int(par[3]*np.random.randint(0,1000000000))
 
    ppp = './BSE/bse_mod << EOF\nbinary_%s.out\n'%(rand_file_name)
    ppp+='%s %s 4000. %s 1 1 0.02 %s\n'%(par[0],par[1],par[2],par[3])   
    ppp+='0.5 %s 1.0 3.0 0.5\n'%(par[4]) # Bw
    ppp+='0 1 0 1 0 1 3.0 29769\n'
    ppp+='0.05 0.01 0.02\n'
    ppp+='190.0 0.125 1.0 1.5 0.001 10.0 -1.0\n'
    ppp+='EOF'     
    timeout_sec =  3
        
    result, flag = run_command_with_timeout(ppp, timeout_sec)
 
    
    skip_header = 0
    skip_footer = 0

    Time=  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [0])        
    M_A =  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [3])
    M_B =  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [4])        
 
    L_B =  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [12]) # in log10                    
 
    R_B =  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [8])  # in log10        
    
    Per =  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [17])  # in yr.   
    ecc =  np.genfromtxt('binary_%s.out'%(rand_file_name),skip_header=skip_header, unpack=True,skip_footer=skip_footer, usecols = [19])   


    os.system("rm -r binary_%s.out"%(rand_file_name))  
    
    L_B =  10.0**L_B                     
 
    R_B =  10.0**R_B   
    
    Per =  Per*365.25  
    

    loglik_arr = [] #np.zeros(len(M_A)):
    for i in range(len(M_A)):
        t_c = np.array([M_A[i],M_B[i],Per[i],ecc[i], R_B[i], L_B[i],Time[i] ])
        loglik_arr.append( -0.5*(np.sum(np.divide((best_fit_est - t_c)**2,(e_best_fit_est**2))) )  )
        
    k = argmax(loglik_arr)

 
   
    loglik = -0.5*(np.sum(np.divide((best_fit_est - [M_A[k],M_B[k],Per[k],ecc[k], R_B[k], L_B[k],Time[k] ])**2,(e_best_fit_est**2))) )
    
    f=open("loglik","a")
    f.write("%s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s\n" %(loglik, par[0],par[1],par[2],par[3], Time[k], M_A[k],M_B[k],Per[k],ecc[k],R_B[k], L_B[k],Time[k],par[4] )) # last param is Bw

    f.close() 

    return loglik
 

 


###########################################################################################
#------------------------- Sort out only those that will be optimized --------------------#
p,f,b,e = concentrate_par(par,flag,bounds,el_str)

 
p = []  #'p" are the fitted parameters
b = []
e = []

for j in range(len(par)):
    if flag[j] > 0:
        p.append(par[j])
        b.append(bounds[j])
        e.append(el_str[j])

####### find the -LogLik "minimum" using the "truncated Newton" method ######### 
nll = lambda *args: -BSE(*args)

minimzers = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B',
'TNC','COBYLA','SLSQP','dogleg','trust-ncg']

for k in range(0): # run at least 3 times the minimizer

    result = op.minimize(nll, p, args=(par,f,best_fit_est, e_best_fit_est), method=minimzers[6],bounds=b, options={'xtol': 1e-6, 
'disp': True })

    p = result["x"]
    print("Best fit par.:", p)
#----------------- one more time using the Simplex method ---------------------#

for k in range(0): # run at least 3 times the minimizer
    result = op.minimize(nll, result["x"], args=(par,f,best_fit_est, e_best_fit_est), method=minimzers[0], options={'xtol': 1e-6, 'disp': True,
 'maxiter':30000, 'maxfev':30000 })


################################################################################ 

best_fit_par = p# result["x"]
print("Best fit par.:", best_fit_par)



####################### Start the mcmc using "emcee"   ######################### 

#------------------------- flat prior for now!!! ------------------------------#
b = np.array(b)


def lnprior(p): 
    for j in range(len(p)):
 
 
        ######## if something is away from the borders - reject #####
        if p[j] <= b[j,0] or p[j] >= b[j,1]:
            return -np.inf
    return 0.0 # + prob_ecc1 + prob_ecc2
  

####################################################### 

def lnprob(p, stmass):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + BSE(p, par,f,best_fit_est, e_best_fit_est)

 

from pathos.pools import ProcessPool as Pool
 
#pool=Pool(ncpus=20)


################## prior TESTS ########################

def prior_transform(p): 
 
        u_trans = np.zeros(len(p)) 
        for jj in range(len(p)): 

                #            if priors[0][j,2] == True:
                #                u_trans[j] = trans_norm(p[j],bb[j][0],bb[j][1])
                #            elif priors[1][j,2] == True:
                #                u_trans[j] = trans_loguni(p[j],bb[j][0],bb[j][1]) 
                #            else:
                u_trans[jj] = trans_uni(p[jj],b[jj][0],b[jj][1])
        return u_trans   


def trans_norm(p ,mu,sig):
    return stats.norm.ppf(p,loc=mu,scale=sig)

def trans_uni(p,a,b):
    return a + (b-a)*p

def trans_loguni(p,a,b):
    return np.exp(np.log(a) + p*(np.log(b)-np.log(a)))    
    



partial_func = FunctionWrapper(BSE, (par,f, best_fit_est,  e_best_fit_est ) )



ndim, nwalkers = len(p), len(p)*7500
 
print_progress = True
Dynamic_nest = True
threads = 40
stop_crit = 0.1



if Dynamic_nest == False:
        print("'Static' Nest. Samp. is running, please wait...")

        if threads > 1:
            with Pool(processes=threads) as thread:
                sampler = dynesty.NestedSampler(partial_func, prior_transform, ndim, nlive=nwalkers, pool = thread, 
                                                queue_size=threads, bootstrap=0)

                sampler.run_nested(dlogz=0.1, print_progress=print_progress)
        else:
                sampler = dynesty.NestedSampler(partial_func, prior_transform, ndim, nlive=nwalkers)
                sampler.run_nested(dlogz=stop_crit, print_progress=print_progress)

        thread.close() 
        thread.join() 
        thread.clear() 
 

else:
        print("'Dynamic' Nest. Samp. is running, please wait... ")

        if threads > 1:
            with Pool(ncpus=threads) as thread:
                sampler = dynesty.DynamicNestedSampler(partial_func, prior_transform, ndim, pool = thread,
                                                       queue_size=threads, sample = 'rwalk', bound = 'multi') # nlive=nwalkers,

                sampler.run_nested(print_progress=True,dlogz_init=0.01,nlive_init=nwalkers, 
                maxiter = 50000000, maxcall = 50000000,use_stop = True, wt_kwargs={'pfrac': 1.0})   #nlive_batch=1
        else:
             sampler = dynesty.DynamicNestedSampler(partial_func, prior_transform, ndim )
             sampler.run_nested(nlive_init=nwalkers, print_progress=print_progress)        

        thread.close() 
        thread.join() 
        thread.clear() 
 

# print("--- %s seconds ---" % (time.time() - start_time))  


weighted =  np.exp(sampler.results.logwt - sampler.results.logz[-1])
samples  =  dill.copy(dynesty.utils.resample_equal(sampler.results.samples, weighted))

ln = np.hstack(sampler.results.logl)
samples = np.array(samples)
    

fileoutput = True
if (fileoutput):
# start_time = time.time()   
# print("Please wait... writing the ascii file")  

        outfile = open(str("nest_sampl"), 'w') # file to save samples
        for j in range(len(samples)):
            outfile.write("%s  " %(ln[j]))
            for z in range(len(p)):
                outfile.write("%s  " %(samples[j,z]))
            outfile.write("\n")
        outfile.close()        
# print("--- Done for ---")           
#   print("--- %s seconds ---" % (time.time() - start_time))  

############### find uncertainties form the result distribution#################

#----------------------------------- labels  ----------------------------------#
 

level = (100.0-68.3)/2.0
best_fit_par_2 = []
print("Best fit par. and their 1 sigma errors" )	
for i in range(len(best_fit_par)):
    ci = np.percentile(samples[:,i], [level, 100.0-level])
    print(e[i],'=', best_fit_par[i], "- %s"%(best_fit_par[i]-ci[0]), "+ %s"%(ci[1]  - best_fit_par[i] ))

print("   ")
print("   " 	)
print("Means and their 1 sigma errors" )	
for i in range(len(best_fit_par)):
    ci = np.percentile(samples[:,i], [level, 100.0-level])
    print(e[i],'=', np.mean(samples[:,i]), "- %s"%(np.mean(samples[:,i])-ci[0]), "+ %s"%(ci[1]  - np.mean(samples[:,i]) ))

    best_fit_par_2.append(np.median(samples[:,i]))



#fig = corner.corner(samples, labels=e, truths=best_fit_par, dpi = 300 )
#fig.savefig("samples_%s.png"%mod)


stop = timeit.default_timer()
#range=ranged, 
fig = corner.corner(samples,bins=25, color="k", reverse=False, upper= True, labels=e, 
                    quantiles=[0.1585, 0.8415],levels=(0.6827, 0.9545,0.9973), smooth=1.0, 
                    smooth1d=1.0, plot_contours= True, show_titles=True, truths=best_fit_par_2, dpi = 300, 
                    pad=15, labelpad = 0 ,truth_color ='r', title_kwargs={"fontsize": 12}, scale_hist=True,  
                    no_fill_contours=True, plot_datapoints=True)


#fig = corner.corner(samples, labels=el_str, truths=best_fit_par, dpi = 300, pad=15, labelpad = 50 )
fig.savefig("samples_%s.pdf"%mod)


#os.system("sort nest_sampl | uniq > nest_sampl_sorted &")
print('Time: ', stop - start)

print("Done")
















