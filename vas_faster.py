import numpy as np
import time
import numba
#import humanize
#metric = humanize.metric

# Lazy metric function in case you dont want to download humanize
def metric(value, unit='', precision=3):
    metric_log = np.log(abs(value))/np.log(1000)
    metric_power = int(np.floor(metric_log))
    tens_power = int(np.floor(metric_log*3))%3
    metric_scale = np.power(1000.0, -metric_power)
    tens_scale = np.power(10.0, tens_power)
    metric_val = value*metric_scale
    tens_val = metric_val/tens_scale
    
    formatter = '%%.%df'%(precision-1)
    tens_val = tens_scale*float(formatter%tens_val)
    
    width = 1+tens_power
    prec = max(precision - width, 0)
    formatter = '%%%d.0%df'%(width, prec)
    str_value = formatter%tens_val
    
    pos_prefixes = "Q R Y Z E P T G M k".split()
    neg_prefixes = "m u n p f a z y r q".split()
    neg_prefixes[1] = chr(956) # If unicode is allowed, use mu
    pref = list(reversed(pos_prefixes + [''] + neg_prefixes))
    if 0 <= metric_power + len(neg_prefixes) < len(pref):
        str_unit = pref[metric_power + len(neg_prefixes)]+unit
    else:
        str_unit = 'e%d'+unit
    return (str_value + ' ' + str_unit).strip()
    
# Timer function usable with a with directive.
# with Timer():
#     ...
class Timer:
    def fmttime(t0_ns, t1_ns):
        return metric((t1_ns-t0_ns)*1e-9, 's')
    def __init__(self, prefix="", suffix="in %s", stats=None, suppress=False):
        self.prefix, self.suffix = prefix, suffix
        self.t0, self.t1 = None, None
        self.stats = stats
        self.suppress = suppress
    def __str__(self):
        if self.t0 is None:
            s = Timer.fmttime(0, 0)
        else:
            s = Timer.fmttime(self.t0, self.t1)
        
        if self.prefix != "":
            s = ' '.join([self.prefix, self.suffix%s])
        return s
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
    def __exit__(self, *args):
        self.t1 = time.time_ns()
        if self.stats is not None:
            self.stats.append((self.t1-self.t0)*1e-9)
        if not self.suppress:
            print(str(self))


@numba.njit
def torgen(N,p,q):
    step=2*np.pi/(N)
    th=np.arange(0,2*np.pi,step)
    r=np.cos(th*q)+2
    x=4*r*np.cos(p*th)
    y=4*r*np.sin(p*th)
    z=-4*np.sin(q*th);
    knot=np.zeros((N,3))
    knot[:,0]=np.transpose(x)
    knot[:,1]=np.transpose(y)
    knot[:,2]=np.transpose(z)
    return knot

@numba.njit # nopython jit
def vas(knot, closed):
    N=len(knot)
    wmat=np.zeros((N,N))
    #k2=np.roll(knot, 1, axis=0)
    #k3=np.roll(knot, -1, axis=0)
    #k23 = np.pad(knot, ((1,1),), mode='wrap')
    #k2, k3 = k23[:-1], k23[1:]

    # njit is picky
    k2 = np.empty_like(knot)
    k2[0], k2[1:] = knot[-1], knot[:-1]
    
    k3 = np.empty_like(knot)
    k3[:-1], k3[-1] = knot[1:], knot[0]
    
    dk=(k2-k3)/2
    if closed==0:
        dk[-1,:]=knot[-1,:]-knot[-2,:]
        dk[0,:]=k2[0,:]-knot[0,:]
    for j in range(0,N):
        for i in range(j+1,N):
            #if i>j:
            dknot_ij = knot[i]-knot[j]
            wmat[i,j]=np.dot(np.cross(dk[i],dk[j]),dknot_ij)/np.linalg.norm(dknot_ij)**3
    SLL=0
    for i in range(3,N):
        for j in range(2,i):
            for k in range(1,j):
                t1=wmat[i,k]
                for l in range(0,k):
                    t2=wmat[j,l]
                    SLL=SLL+t1*t2/(8*np.pi)
    v2=SLL/np.sqrt(12*np.pi)
    return v2

# stats = []
# with Timer(stats=stats):
#     print(vas(torgen(100,3,2),1))

# for i in range(10):
#     with Timer(stats=stats):
#         print(vas(torgen(100,3,2),1))

# print(metric(np.mean(stats[1:]), 's'), chr(177), metric(np.std(stats[1:]), 's'))

# knot=torgen(100,3,2)

# print(vas(knot, 1), 'value of vassiliev')

# import matplotlib.pyplot as plt
# ax = plt.figure().add_subplot(projection='3d')
# x = knot[:,0]
# y = knot[:,1]
# z = knot[:,2]
# ax.plot(x, y, z)
# plt.show()
# ax.show()