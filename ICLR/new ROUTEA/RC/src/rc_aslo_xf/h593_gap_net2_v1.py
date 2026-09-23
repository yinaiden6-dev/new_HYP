"""Direct net-gain search with unchanged m + beta*d + bias HOLD model.

Searches a deterministic finite set of training-line crossings and interval
representatives, plus the incumbent. Exact conditional bias maximization is
performed at each sampled slope. No global binary64-plane optimum is claimed.
"""
import hashlib
import math

import numpy as np

from rc_aslo_xf.h593_s_bias_competition_v1 import calibrate_bias, apply_head


def arrays(records):
    rows = [r for r in records if float(r['m']) <= 0.]
    m = np.array([r['m'] for r in rows], dtype=np.float64)
    d = np.array([r['d'] for r in rows], dtype=np.float64)
    y = np.array([r['delta'] for r in rows], dtype=np.int64)
    if not np.isfinite(m).all() or not np.isfinite(d).all() or np.any(d < 0) or not np.isin(y,[-1,0,1]).all():
        raise ValueError('invalid calibration records')
    return m, d, y


def digest(array):
    return hashlib.sha256(np.asarray(array,dtype='<f8').tobytes(order='C')).hexdigest()


def slope_candidates(m, d, incumbent_beta):
    """Training-defined finite slope set; no outer labels or handpicked range."""
    if not math.isfinite(incumbent_beta) or incumbent_beta < 0:
        raise ValueError('invalid incumbent slope')
    i,j = np.triu_indices(len(m), 1)
    denominator = d[i]-d[j]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        crossing = (m[j]-m[i])/denominator
    crossing = np.unique(crossing[np.isfinite(crossing)&(crossing>0)])
    bases = np.concatenate(([0.],crossing))
    middle = bases[:-1] + (bases[1:]-bases[:-1])/2.
    before = np.nextafter(crossing, -np.inf)
    after = np.nextafter(crossing, np.inf)
    last = float(crossing[-1]) if len(crossing) else 0.
    # Explicit representative of the last real-line ordering interval.
    with np.errstate(over='ignore'):
        tail = last + max(1.,abs(last))
    values = np.concatenate(([0.,incumbent_beta], crossing, before, after, middle, [tail]))
    values = values[np.isfinite(values)&(values>=0)]
    values[values==0] = 0.
    return np.unique(values)


def scan(m, d, y, betas, batch_size=256):
    """FP64 independent operations followed by sorted, tie-grouped prefix sums.

Return [net, changes, rescues, breaks, neutral, closest_zero_bias] per slope.
The explicit disabled action is represented by all zero metrics/bias.
"""
    n = len(m)
    table = np.zeros((len(betas),6),dtype=np.float64)
    if n == 0:return table
    size = np.arange(1,n+1,dtype=np.int64)[None,:]
    for start in range(0,len(betas),batch_size):
        bs = betas[start:start+batch_size]
        with np.errstate(over='ignore',invalid='ignore'):
            u = np.add(np.add(m,0.)[None,:], np.multiply(bs[:,None],d[None,:]))
        if not np.isfinite(u).all():raise ValueError('nonfinite candidate offsets')
        order = np.argsort(-u,axis=1,kind='stable')
        us = np.take_along_axis(u,order,axis=1)
        yy = y[order]
        gain = np.cumsum(yy,axis=1)
        valid_end = np.concatenate((us[:,:-1]!=us[:,1:],np.ones((len(bs),1),dtype=bool)),axis=1)
        encoded = np.where(valid_end,gain*(n+1)-size,-10*(n+1)**2)
        k = np.argmax(encoded,axis=1);idx=np.arange(len(bs))
        nets = gain[idx,k];positive=nets>0
        # first positive transition and last bias before the next transition
        lower=np.nextafter(-us[idx,k],np.inf)
        upper=np.full(len(bs),np.finfo(np.float64).max)
        has_next=k+1<n
        upper[has_next]=-us[idx[has_next],k[has_next]+1]
        bias=np.where(lower>0,lower,np.where(upper<0,upper,0.))
        if not np.isfinite(bias[positive]).all():raise ValueError('nonfinite selected bias')
        rescues=np.cumsum(yy==1,axis=1)[idx,k]
        breaks=np.cumsum(yy==-1,axis=1)[idx,k]
        values=np.column_stack((nets,k+1,rescues,breaks,k+1-rescues-breaks,bias))
        values[~positive]=0.
        # Literal replay of the selected pattern guards order/tie shortcuts.
        with np.errstate(over='ignore',invalid='ignore'):
            switched=np.add(u,bias[:,None])>0
        if not np.array_equal(np.sum(switched*y,axis=1)[positive],nets[positive]):
            raise ArithmeticError('net prefix replay failed')
        if not np.array_equal(np.sum(switched,axis=1)[positive],(k+1)[positive]):
            raise ArithmeticError('tie prefix replay failed')
        table[start:start+len(bs)]=values
    return table


def row_counts(records, head):
    selected=[]
    beta=float.fromhex(head['beta_hex']);bias=float.fromhex(head['bias_hex'])
    for r in records:
        if r['m']<=0 and not head['disabled']:
            score=float(float(float(r['m'])+0.)+float(beta*r['d']))+bias
            if score>0:selected.append(int(r['delta']))
    return dict(net=sum(selected),changes=len(selected),rescues=selected.count(1),
                breaks=selected.count(-1),neutral=selected.count(0))


def choose(betas, table, incumbent_beta, incumbent_net):
    best = int(np.max(table[:,0],initial=0.))
    if best <= incumbent_net:return None
    eligible=np.flatnonzero(table[:,0]==best)
    def key(i):
        beta=float(betas[i]);b=float(table[i,5])
        return (int(table[i,1]),abs(math.log1p(beta)-math.log1p(incumbent_beta)),abs(b),beta,b)
    return min(map(int,eligible),key=key)


def fit_direct(records, incumbent):
    if float.fromhex(incumbent['alpha_hex']) != 0.:raise ValueError('gap-only incumbent required')
    beta0=float.fromhex(incumbent['beta_hex'])
    old=row_counts(records,incumbent)
    if old['net']!=incumbent['training_net_gain']:raise ValueError('incumbent calibration mismatch')
    m,d,y=arrays(records);betas=slope_candidates(m,d,beta0);table=scan(m,d,y,betas)
    index=choose(betas,table,beta0,old['net'])
    # Retain actual incumbent parameters on every non-improving training tie.
    if index is None:
        head={k:incumbent[k] for k in ('alpha','beta','bias','alpha_hex','beta_hex','bias_hex','disabled')}
    else:
        head=calibrate_bias(records,0.,float(betas[index]))
        counts=row_counts(records,head)
        assert [counts[k] for k in ('net','changes','rescues','breaks','neutral')]==list(map(int,table[index,:5]))
        assert head['bias_hex']==float(table[index,5]).hex()
    counts=row_counts(records,head)
    if counts['net']<old['net']:raise ArithmeticError('calibration incumbent regression')
    head.update(mode='GAP_NET2',training_net_gain=counts['net'],training_changed_holds=counts['changes'],
                training_rescues=counts['rescues'],training_breaks=counts['breaks'],training_neutral=counts['neutral'],
                search=dict(status='FINITE_TRAINING_SLOPE_SET_NET_OPTIMUM',candidate_count=len(betas),
                            slope_sha256=digest(betas),table_sha256=digest(table),
                            incumbent_beta_hex=incumbent['beta_hex'],incumbent_bias_hex=incumbent['bias_hex'],
                            incumbent_counts=old,selected_index=index,incumbent_retained=index is None,
                            best_candidate_net=int(np.max(table[:,0],initial=0.)),
                            selected_counts=counts,query_count=len(records),original_hold_count=len(m),
                            full_binary64_plane_optimum_claimed=False,
                            tie_rule='strict net improvement over incumbent; then fewer changes, closest log1p beta to incumbent, smallest abs bias, beta, bias'))
    return head
