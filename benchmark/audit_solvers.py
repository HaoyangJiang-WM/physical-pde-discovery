"""Independent DOP853 forward checks, timestep/spatial checks, sensitivity checks."""
from experiment import *
from scipy.integrate import solve_ivp

def independent(case,th,setting):
    a,b=th[:2];c=setting[0]
    if case=='reaction':ic=[c];fun=lambda t,z:-a*z-b*z*z
    elif case=='reaction3':ic=[c];fun=lambda t,z:-a*z-b*z*z-th[2]*z**3
    elif case=='chemostat':ic=[1.];fun=lambda t,z:-(a+b*c)*z
    elif case=='reversible':ic=[c];fun=lambda t,z:-a*z+b*(1-z)
    elif case=='enzyme':ic=[c];fun=lambda t,z:-a*z-b*z/(.4+z)
    elif case=='oscillator':
        ic=[c,0.]
        fun=lambda t,z:np.array([-z[1]+(a-b*np.sum(z*z))*z[0],z[0]+(a-b*np.sum(z*z))*z[1]])
    elif case=='duffing':ic=[c,0.];fun=lambda t,z:np.array([z[1],-.3*z[1]-a*z[0]-b*z[0]**3])
    elif case=='forced_damping':ic=[0.,0.];fun=lambda t,z:np.array([z[1],-z[0]-a*z[1]-b*z[1]*abs(z[1])+2*np.sin(c*np.pi*t/T)])
    elif case=='predator':ic=list(setting);fun=lambda t,z:np.array([.5*z[0]-a*z[0]*z[1],b*z[0]*z[1]-.4*z[1]])
    elif case=='diffusion_pde':ic=[1.];fun=lambda t,z:-(a*c*c+b*c**4)*z
    elif case=='advection_diffusion':ic=[1.,0.];fun=lambda t,z:np.array([-b*c*c*z[0]-a*c*z[1],a*c*z[0]-b*c*c*z[1]])
    else:
        nx=32;xx=2*np.pi*np.arange(nx)/nx;ic=c*np.cos(setting[1]*xx);dx=2*np.pi/nx
        fun=lambda t,z:.04*(np.roll(z,1)-2*z+np.roll(z,-1))/dx**2+a*z-b*z**3
    sol=solve_ivp(fun,[0,T],ic,t_eval=TIMES,rtol=2e-11,atol=2e-12,method='DOP853')
    if not sol.success:raise RuntimeError(sol.message)
    y=sol.y.T
    if case=='oscillator':return np.sum(y*np.column_stack([np.cos(TIMES),np.sin(TIMES)]),axis=1)
    if case=='allen_cahn':return y[:,::4].ravel()
    return y.ravel()

def main():
    rng=np.random.default_rng(1498101);rows=[]
    for case in CASES+['reaction3']:
        rr=RANGES[case]
        for i in range(12):
            th=rng.uniform(rr[:,0],rr[:,1]);th[rng.random(len(th))<.35]=0
            ac=MENUS[case][i%5]
            model=response(case,th,ac)[0];ref=independent(case,th,ac)
            fine=response(case,th,ac,substeps=32)[0]
            J=jacobian(case,th,ac);cols=[]
            for j in range(len(th)):
                h=5e-6*rr[j,1];tp=th.copy();tm=th.copy();tp[j]+=h;tm[j]-=h
                cols.append((response(case,tp,ac,substeps=32)[0]-response(case,tm,ac,substeps=32)[0])/(2*h))
            Jref=np.stack(cols,axis=1)
            row=dict(case=case,trial=i,dop853_max=float(np.abs(model-ref).max()),dt_half_max=float(np.abs(model-fine).max()),jac_rel=float(np.linalg.norm(J-Jref)/max(1e-9,np.linalg.norm(Jref))))
            if case=='allen_cahn':
                y64=response(case,th,ac,substeps=32,nx=64)[0];y128=response(case,th,ac,substeps=64,nx=128)[0]
                row.update(spatial32_64=float(np.abs(model-y64).max()),spatial64_128=float(np.abs(y64-y128).max()))
            rows.append(row)
        print(case,'audited',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'audits'/'solver.csv',index=False)
    print(pd.DataFrame(rows).groupby('case').max(numeric_only=True).to_string())
if __name__=='__main__':main()
