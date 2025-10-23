import numpy as np


def my_levy_stable(alpha=1.0, beta=0.0, loc=0.0, scale=1.0):
    mu, sigma = loc, scale
    V = np.pi * np.random.random() - np.pi/2
    W = -np.log(1 - np.random.random())
    if alpha == 1:
        dzeta = mu + 2 * sigma * ((np.pi/2 + beta*V) * np.tan(V) - 
                beta * np.log((np.pi * W * np.cos(V)/2)/(np.pi/2 + beta*V))) / np.pi
    else:
        C = np.arctan(beta * np.tan(np.pi * alpha/2)) / alpha
        D = sigma * (np.cos(np.arctan(beta * np.tan(np.pi * alpha/2))))**(-1/alpha)
        dzeta = D * np.sin(alpha*(V + C))*(np.cos(V - alpha*(V + C))/W)**(1/alpha - 1) / np.cos(V)**(1/alpha)
    return dzeta


class IonChannelModel:
    state = ['open', 'close']
    
    def __init__(self, states={}, full_stats=False):
        self.full_stats = full_stats
        self.statistics = []
        self.states = states
        
    @property
    def states(self):
        return self.__states
    
    @states.setter
    def states(self, s):
        assert isinstance(s, dict), 'states should be a dictionary'
        self.__states = s
        
        if 'open' not in self.states:
            self.__states['open'] = [1]
            
        if 'open value' not in self.states:
            self.__states['open value'] = 1
        
        if 'close' not in self.states:
            self.__states['close'] = [1]
            
        if 'close value' not in self.states:
            self.__states['close value'] = -1
        
        # all states
        len_states = len(self.states['open']) + len(self.states['close'])
        
        if 'scheme' not in self.states:
            # first open, then close, no loops
            self.__states['scheme'] = [[0 if i==j else 1 for j in range(len_states)] for i in range(len_states)]
            self.__states['scheme choice'] = 'all allowed'
        else:
            self.__states['scheme choice'] = 'user'
            if self.states['scheme'] == 'random':
                self.__states['scheme choice'] += ' random'
                self.random_scheme()
                
            if (len(self.states['scheme']) != len_states 
                or
                len(self.states['scheme'][0]) != len_states):
                raise ValueError('scheme dimentions error')
        
        if 'inital state' not in self.states:
            self.__states['initial state'] = self.next_state()
    
    def generate_random_scheme(self):
        N = len(self.states['open']) + len(self.states['close'])
        M = np.random.randint(0, 2, (N, N))
        for idx in range(N):
            for jdx in range(N):
                if idx == jdx:
                    M[idx][jdx] = 0
        return M
    
    def scheme_ok(self, M):
        '''
        at least one 1 should be in each column AND row;
        mind the zeros on diagonal;
        '''
        M = np.asarray(M)
        nrows, ncols = M.shape
        for irow in range(nrows):
            if not any(M[irow, :]):
                return False
        for icol in range(ncols):
            if not any(M[:, icol]):
                return False
        return True
    
    def random_scheme(self):
        '''returns rate matrix NxN'''
        N = len(self.states['open']) + len(self.states['close'])
        M = np.zeros((N, N))
        
        while not self.scheme_ok(M):
            M = self.generate_random_scheme()
            
        self.__states['scheme'] = M
    
    @property
    def full_stats(self):
        return self.__full_stats
    
    @full_stats.setter
    def full_stats(self, val):
        assert isinstance(val, bool), 'full_stats: True/False'
        self.__full_stats = val
    
    @property
    def statistics(self):
        return self.__statistics
    
    @statistics.setter
    def statistics(self, val):
        self.__statistics = val

    def update_stats(self, val):
        self.statistics.append(val)
    
    @staticmethod
    def draw_dwell_time(tau=1.0):
        """
        Generate a random dwell time sampled from an exponential distribution.

        Parameters:
            tau (float, optional): The mean (scale parameter) of the exponential distribution.
                Defaults to 1.0.

        Returns:
            float: A random dwell time value sampled from the exponential distribution with mean `tau`.
        """
        return np.random.exponential(tau)

    @staticmethod
    def pretty_print(L):
        r = ''
        for el in L:
            r += str(el)[1:-1].replace(',','') + '\n'
        return r
    
    def __repr__(self):
#         r = f"open states {self.states['open']}\n"
#         r += f"close states {self.states['close']}\n"
#         r += f"scheme\n{__class__.pretty_print(self.states['scheme'])}"
        r = str(self.states)
        return r
    
    def scheme_index(self, state, tau_hat):
        border = len(self.states['open'])
        if state == 'open':
            idx = self.states['open'].index(tau_hat)
        else:
            idx = border + self.states['close'].index(tau_hat)
        return idx
    
    def verboten_transition(self, prev_state, prev_tau_hat, new_state, new_tau_hat):
        border = len(self.states['open'])
        prev_idx = self.scheme_index(prev_state, prev_tau_hat)            
        new_idx = self.scheme_index(new_state, new_tau_hat)
        rate = self.states['scheme'][new_idx][prev_idx]
        # print(prev_state, prev_tau_hat, new_state, new_tau_hat, rate)
        return not bool(rate)
    
    def next_state(self, prev_state=None, prev_tau_hat=None):
        # TODO: use rates provided in scheme
        if prev_state == None:
            new_state = np.random.choice(__class__.state)
            new_tau_hat = np.random.choice(self.states[new_state])
            return new_state, new_tau_hat
        
        new_state, new_tau_hat = prev_state, prev_tau_hat
        while self.verboten_transition(prev_state, prev_tau_hat, new_state, new_tau_hat):
            #print('Verboten!', end='___')
            new_state = np.random.choice(__class__.state)
            new_tau_hat = np.random.choice(self.states[new_state])
        
        return new_state, new_tau_hat
    
    #
    # sections below are for defining forces and potentials
    #
    def vq(self, x, params={'xs': 0, 'a': 1, 'd': 0}):
        xs = params.get('xs', 0)
        a = params.get('a', 1)
        d = params.get('d', 0)
        return a*(x-xs)**2 - d  # simple quadratic

    def fq(self, x, params={'xs': 0, 'a': 1}):
        xs = params.get('xs', 0)
        a = params.get('a', 1)
        return -2*a*(x-xs) # simple quadratic

    def vm(self, x, params={'xs': 0, 'a': 1, 'b': 1, 'd': 0}):
        xs = params.get('xs', 0)
        a = params.get('a', 1)
        b = params.get('b', 1)
        d = params.get('d', 0)
        return np.where(x < xs, 
                        self.vq(x, {'xs': xs, 'a': a, 'd': d}), 
                        self.vq(x, {'xs': xs, 'a': b, 'd': d}))  
                        # modified quadratic on both sides

    def fm(self, x, params={'xs': 0, 'a': 1, 'b': 1}):
        xs = params.get('xs', 0)
        a = params.get('a', 1)
        b = params.get('b', 1)
        return np.where(x < xs, 
                        self.fq(x, {'xs': xs, 'a': a}), 
                        self.fq(x, {'xs': xs, 'a': b}))  
                        # modified quadratic on both sides

    def vp(self, x, params={'xs': 0, 'a': 1, 'b': 1, 'd': 0}):
        xs = params.get('xs', 0)
        a = params.get('a', 1)
        b = params.get('b', 1)
        d = params.get('d', 0)
        return np.where(x > xs, 
                        self.vq(x, {'xs': xs, 'a': a, 'd': d}), 
                        self.vq(x, {'xs': xs, 'a': b, 'd': d}))  
                        # modified quadratic on both sides

    def fp(self, x, params={'xs': 0, 'a': 1, 'b': 1}):
        xs = params.get('xs', 0)
        a = params.get('a', 1)
        b = params.get('b', 1)
        return np.where(x > xs, 
                        self.fq(x, {'xs': xs, 'a': a}), 
                        self.fq(x, {'xs': xs, 'a': b}))
                        # modified quadratic on x>0

    def vb(self, x, params={'xs': 0, 'b': 1, 'c': 1, 'd': 0}):
        xs = params.get('xs', 0)
        b = params.get('b', 1)
        c = params.get('c', 1)
        d = params.get('d', 0)
        return b * (x-xs) ** 4 / 4 - c * (x-xs) ** 2 / 2 + d*x  # double well

    def potential_asym_quadratic(self, x, 
                                 params={'xs': 0, 
                                         'aplus': 1, 'bplus': 1, 'dplus': 0, 
                                         'aminus': 1, 'bminus': 1, 'dminus': 0}):
        xs = params.get('xs', 0)
        aplus = params.get('aplus', 1)
        bplus = params.get('bplus', 1)
        dplus = params.get('dplus', 0)
        aminus = params.get('aminus', 1)
        bminus = params.get('bminus', 1)
        dminus = params.get('dminus', 0)
        return np.where(x > 0,
                        self.vp(x, {'xs': xs, 'a': aplus, 'b': bplus, 'd': dplus}),
                        self.vm(x, {'xs': xs, 'a': aminus, 'b': bminus, 'd': dminus}))

    def force_asym_quadratic(self, x, 
                             params={'xs': 0, 'aplus': 1, 'bplus': 1, 'aminus': 1, 'bminus': 1}):
        xs = params.get('xs', 0)
        aplus = params.get('aplus', 1)
        bplus = params.get('bplus', 1)
        aminus = params.get('aminus', 1)
        bminus = params.get('bminus', 1)
        return np.where(x > 0, 
                        self.fp(x, {'xs': xs, 'a': aplus, 'b': bplus}), 
                        self.fm(x, {'xs': xs, 'a': aminus, 'b': bminus}))

    def force_bistable(self, x, params={'a': 1, 'b': 1}):
        '''double well
        V = a / 4 * x**4 - b / 2 * x**2'''
        a = params.get('a', 1)
        b = params.get('b', 1)
        return - a * x**3 + b * x

    def potential_bistable(self, x, params={'a': 1, 'b': 1}):
        a = params.get('a', 1)
        b = params.get('b', 1)
        return a / 4 * x**4 - b / 2 * x**2

    def force_x2(self, x, params={'a': 0, 'b': 1}):
        '''monostable at x=a
        V = b / 2 * (x - a)**2'''
        a = params.get('a', 0)
        b = params.get('b', 1)
        return -b * (x - a)

    def potential_x2(self, x, params={'a': 0, 'b': 1}):
        a = params.get('a', 0)
        b = params.get('b', 1)
        return b / 2 * (x - a)**2

    def force_const(self, x, params={'xs': 1}):
        xs = params.get('xs', 1)
        return -xs

    def potential_const(self, x, params={'xs': 1}):
        xs = params.get('xs', 1)
        return xs * x

    def force(self, x, params={'a': 1, 'b': 1}):
        '''potential'''
        return self.force_x2(x, params)

    def noise(self, rng, params={'a': 1}, levy_stat=True):
        a = params.get('a', 1)
        if levy_stat:
            return my_levy_stable(1.25, 0, loc=a)
        return rng.standard_normal()
    
    def simulate(self, tend=1, h=0.01, D=0.01, levy_stat=False, force=None):
        rng = np.random.default_rng()
        
        if force == None:
            raise ValueError('No force provided')
        
        if 'potential' not in force:
            raise ValueError('No force type provided')
        
        if 'params' not in force:
            raise ValueError('No force params provided')
        force_params = force['params']

        if force['potential'] == 'x2':
            F = self.force_x2

        elif force['potential'] == 'bistable':
            raise NotImplementedError('Bistable dynamics not implemented')
            F = self.force_bistable

        elif force['potential'] == 'const':
            F = self.force_const

        elif force['potential'] == 'asym_quadratic':
            F = self.force_asym_quadratic
        
        state, tau_hat = self.states['initial state']
        tau = __class__.draw_dwell_time(tau_hat)

        if self.full_stats:
            self.update_stats((state, tau_hat, tau))

        iks = self.__states[f'{state} value']
        x = [iks]
        t = 0
        while t <= tend:
            if t > tau:
                # print('waaaat?', end='...')
                state, tau_hat = self.next_state(state, tau_hat)

                if tau_hat is None:
                    raise ValueError("tau_hat cannot be None when drawing dwell time")
                next_tau = __class__.draw_dwell_time(tau_hat)
                
                if self.full_stats: 
                    self.update_stats((state, tau_hat, next_tau))
                tau += next_tau
                
                if F.__name__ == self.force_const.__name__:
                    iks = F(0, {'xs': self.__states[f'{state} value']})
                    
                
            else:
                force_params['xs'] = self.__states[f'{state} value']
                if F.__name__ == self.force_const.__name__:
                    iks = F(0, force_params)
                else:
                    iks += h * F(iks, force_params)
                iks += (2 * D * h) ** 0.5 * self.noise(rng, {'a': self.__states[f'{state} value']}, 
                                                       levy_stat=levy_stat)
                           
                x.append(iks)
            t += h

        return x