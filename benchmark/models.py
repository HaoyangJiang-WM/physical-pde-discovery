"""Small sparse dynamical libraries with fixed observation horizons.
All coefficients below are unknown unless explicitly marked known.
No observation at t=0 is included: initial states are controlled inputs.
"""
import numpy as np
from numba import njit

T=1.5
TIMES=np.linspace(0,T,17)[1:]
Q=.4
CASES=['reaction','chemostat','reversible','enzyme','oscillator','duffing','forced_damping','predator','diffusion_pde','advection_diffusion','allen_cahn']
RANGES={
'reaction':[[.05,1.2],[.05,1.2]], 'chemostat':[[.05,1.2],[.05,1.2]],
'reversible':[[.05,1.2],[.05,1.2]], 'enzyme':[[.05,1.2],[.05,1.2]],
'oscillator':[[.05,.7],[.05,.7]], 'duffing':[[.15,1.8],[.05,.8]],
'forced_damping':[[.05,.8],[.05,.8]], 'predator':[[.1,.9],[.1,.9]],
'diffusion_pde':[[.03,.35],[.002,.045]],
'advection_diffusion':[[.05,.8],[.025,.25]],
'allen_cahn':[[.05,.65],[.05,.65]],
'reaction3':[[.05,.7],[.05,.7],[.05,.7]],
}
RANGES={k:np.asarray(v,dtype=float) for k,v in RANGES.items()}
MENUS={
'reaction':[(x,) for x in [.15,.3,.6,1.2,2.]],
'chemostat':[(x,) for x in [.1,.4,1.,1.6,2.5]],
'reversible':[(x,) for x in [.1,.3,.5,.7,.9]],
'enzyme':[(x,) for x in [.1,.25,.5,1.,2.]],
'oscillator':[(x,) for x in [.15,.3,.6,1.1,1.8]],
'duffing':[(x,) for x in [.2,.4,.7,1.,1.4]],
'forced_damping':[(x,) for x in [1.,2.,3.,4.,6.]],
'predator':[(.3,.3),(.3,1.2),(1.2,.3),(.8,.8),(1.2,1.2)],
'diffusion_pde':[(float(k),) for k in range(1,6)],
'advection_diffusion':[(float(k),) for k in range(1,6)],
'allen_cahn':[(.2,1.),(.5,1.),(1.,1.),(.5,2.),(1.,2.)],
'reaction3':[(x,) for x in [.15,.3,.6,1.2,1.8]],
}
INITIAL={
'reaction':(.3,), 'chemostat':(1.,), 'reversible':(.5,), 'enzyme':(.25,),
'oscillator':(.3,), 'duffing':(.4,), 'forced_damping':(2.,),
'predator':(.3,.3), 'diffusion_pde':(1.,),'advection_diffusion':(1.,),
'allen_cahn':(.2,1.), 'reaction3':(.3,),
}
HELDOUT={k:[MENUS[k][0],MENUS[k][2],MENUS[k][4]] for k in MENUS}
HELDOUT.update({
'reaction':[(.45,),(.9,),(1.5,)],'chemostat':[(.25,),(.75,),(2.,)],
'reversible':[(.2,),(.6,),(.8,)],'enzyme':[(.18,),(.7,),(1.5,)],
'oscillator':[(.45,),(.9,),(1.5,)],'duffing':[(.3,),(.85,),(1.2,)],
'forced_damping':[(1.5,),(2.5,),(5.,)],'predator':[(.6,.4),(.4,.9),(1.,.6)],
'diffusion_pde':[(2.,),(3.,),(4.,)],'advection_diffusion':[(2.,),(3.,),(4.,)],
'allen_cahn':[(.35,1.),(.8,2.),(.7,3.)], 'reaction3':[(.45,),(.9,),(1.5,)],
})
EQUATIONS={
'reaction':"x'=-a*x-b*x^2", 'chemostat':"x'=-(a+b*c)*x; c is controlled, x(0)=1",
'reversible':"x'=-a*x+b*(1-x); total concentration known =1",
'enzyme':"x'=-a*x-b*x/(K+x); K=.4 known",
'oscillator':"x'=-y+(a-b*(x^2+y^2))*x; y'=x+(a-b*(x^2+y^2))*y",
'duffing':"x'=v; v'=-.3*v-a*x-b*x^3",
'forced_damping':"x'=v; v'=-x-a*v-b*v*abs(v)+2*sin(n*pi*t/T)",
'predator':"x'=.5*x-a*x*y; y'=b*x*y-.4*y",
'diffusion_pde':"u_t=a*u_xx-b*u_xxxx; u(x,0)=cos(k*x)",
'advection_diffusion':"u_t=-a*u_x+b*u_xx; u(x,0)=cos(k*x)",
'allen_cahn':"u_t=.04*u_xx+a*u-b*u^3; u(x,0)=A*cos(k*x); 32-site periodic MOL",
'reaction3':"x'=-a*x-b*x^2-c*x^3",
}

@njit(cache=True)
def rhs(kind,x,v,t,theta,setting):
    a,b=theta[0],theta[1]
    if kind==0: return -a*x-b*x/(.4+x),0.
    if kind==1: return v,-.3*v-a*x-b*x*x*x
    if kind==2: return v,-x-a*v-b*v*abs(v)+2*np.sin(setting[0]*np.pi*t/1.5)
    if kind==3: return .5*x-a*x*v,b*x*v-.4*v
    return -a*x-b*x*x-theta[2]*x*x*x,0.

@njit(cache=True)
def ode_batch(kind,theta,setting,times,substeps):
    n=len(theta);nt=len(times);nc=1 if kind in (0,4) else 2
    ans=np.empty((n,nt*nc))
    for r in range(n):
        x=setting[0] if kind!=2 else 0.
        v=setting[1] if kind==3 else 0.
        t=0.
        for q in range(nt):
            dt=(times[q]-t)/substeps
            for _ in range(substeps):
                k1,l1=rhs(kind,x,v,t,theta[r],setting)
                k2,l2=rhs(kind,x+dt*k1/2,v+dt*l1/2,t+dt/2,theta[r],setting)
                k3,l3=rhs(kind,x+dt*k2/2,v+dt*l2/2,t+dt/2,theta[r],setting)
                k4,l4=rhs(kind,x+dt*k3,v+dt*l3,t+dt,theta[r],setting)
                x+=dt*(k1+2*k2+2*k3+k4)/6
                v+=dt*(l1+2*l2+2*l3+l4)/6
                t+=dt
            ans[r,q*nc]=x
            if nc==2:ans[r,q*nc+1]=v
    return ans

@njit(cache=True)
def pde_rhs(u,a,b,fac,out):
    n=len(u)
    for j in range(n):out[j]=fac*(u[(j+1)%n]-2*u[j]+u[(j-1)%n])+a*u[j]-b*u[j]**3

@njit(cache=True)
def rd_batch(theta,setting,times,substeps,nx):
    n=len(theta);ans=np.empty((n,len(times)*8));fac=.04/(2*np.pi/nx)**2
    for r in range(n):
        u=setting[0]*np.cos(setting[1]*np.arange(nx)*2*np.pi/nx)
        k1=np.empty(nx);k2=np.empty(nx);k3=np.empty(nx);k4=np.empty(nx);tmp=np.empty(nx)
        t=0.
        for q in range(len(times)):
            dt=(times[q]-t)/substeps
            for _ in range(substeps):
                pde_rhs(u,theta[r,0],theta[r,1],fac,k1)
                for j in range(nx):tmp[j]=u[j]+dt*k1[j]/2
                pde_rhs(tmp,theta[r,0],theta[r,1],fac,k2)
                for j in range(nx):tmp[j]=u[j]+dt*k2[j]/2
                pde_rhs(tmp,theta[r,0],theta[r,1],fac,k3)
                for j in range(nx):tmp[j]=u[j]+dt*k3[j]
                pde_rhs(tmp,theta[r,0],theta[r,1],fac,k4)
                for j in range(nx):u[j]+=dt*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6
                t+=dt
            for j in range(8):ans[r,8*q+j]=u[j*(nx//8)]
    return ans

def response(case,theta,setting,times=TIMES,substeps=16,nx=32):
    theta=np.atleast_2d(np.asarray(theta,dtype=float));setting=np.atleast_1d(setting).astype(float)
    a=theta[:,0,None];b=theta[:,1,None];t=np.atleast_1d(times)[None,:];c=setting[0]
    if case=='chemostat': return np.exp(-(a+b*c)*t)
    if case=='reversible':
        rate=a+b;safe=np.where(abs(rate)<1e-10,1.,rate)
        phi=np.where(abs(rate)<1e-10,t,-np.expm1(-rate*t)/safe)
        return c*np.exp(-rate*t)+b*phi
    if case=='reaction':
        safe=np.where(abs(a)<1e-10,1.,a);phi=np.where(abs(a)<1e-10,t,-np.expm1(-a*t)/safe)
        return c*np.exp(-a*t)/(1+b*c*phi)
    if case=='oscillator':
        safe=np.where(abs(a)<1e-10,1.,a);phi=np.where(abs(a)<1e-10,2*t,np.expm1(2*a*t)/safe)
        return c*np.exp(a*t)/np.sqrt(1+b*c*c*phi)
    if case=='diffusion_pde':return np.exp(-(a*c*c+b*c**4)*t)
    if case=='advection_diffusion':
        amp=np.exp(-b*c*c*t);phase=a*c*t
        return np.stack([amp*np.cos(phase),amp*np.sin(phase)],axis=2).reshape(len(theta),-1)
    if case=='allen_cahn':return rd_batch(theta,setting,np.asarray(times),substeps,nx)
    kind={'enzyme':0,'duffing':1,'forced_damping':2,'predator':3,'reaction3':4}[case]
    return ode_batch(kind,theta,setting,np.asarray(times),substeps)

def jacobian(case,theta,setting):
    theta=np.asarray(theta,dtype=float);cols=[]
    for j in range(len(theta)):
        step=2e-5*RANGES[case][j,1];tp=theta.copy();tm=theta.copy();tp[j]+=step;tm[j]-=step
        cols.append((response(case,tp,setting)[0]-response(case,tm,setting)[0])/(2*step))
    return np.stack(cols,axis=1)
