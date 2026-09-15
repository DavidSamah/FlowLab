"""Inspectable 2-D incompressible-flow building blocks."""
import numpy as np

def advect(field, u, v, dt, dx=1.0):
    """Semi-Lagrangian backtrace with clamped bilinear sampling."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = np.clip(xx - dt*u/dx, 0, w-1); y = np.clip(yy - dt*v/dx, 0, h-1)
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,w-1); y1=np.minimum(y0+1,h-1)
    sx=x-x0; sy=y-y0
    return ((1-sx)*(1-sy)*field[y0,x0] + sx*(1-sy)*field[y0,x1]
            + (1-sx)*sy*field[y1,x0] + sx*sy*field[y1,x1])

def pressure_project(u, v, iterations=80, dx=1.0):
    """Subtract pressure gradient using Jacobi iterations; returns fields and diagnostics."""
    div=(np.gradient(u,dx,axis=1)+np.gradient(v,dx,axis=0))
    p=np.zeros_like(u)
    for _ in range(iterations):
        q=np.pad(p,1,mode="edge")
        p=(q[1:-1,2:]+q[1:-1,:-2]+q[2:,1:-1]+q[:-2,1:-1]-div*dx*dx)/4
    u2=u-np.gradient(p,dx,axis=1); v2=v-np.gradient(p,dx,axis=0)
    return u2,v2,float(np.abs(np.gradient(u2,dx,axis=1)+np.gradient(v2,dx,axis=0)).mean())

def poiseuille_profile(height=32, pressure_gradient=1.0, viscosity=1.0):
    """Analytical planar channel profile u(y)=G/(2ν)y(H-y)."""
    y=np.linspace(0, height-1, height); H=height-1
    return pressure_gradient*y*(H-y)/(2*viscosity)
