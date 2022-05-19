
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.special import lpmv as lpmv
from math import factorial as factorial


n = 2
l = 1
m = -1
n2 = 1
l2 = 0
m2 = 0
#Spherical Harmonic functions
def AssociatedLegendre(l,m,x):
    lpmv(m,l,x) # Annoyingly lpmv needs m to come first in the call order.

    return lpmv(m,l,x)

def AssociatedLaguerre(n,m,x):
    Anm = 0

    for a in range(0,n+1):
        Zeta = (factorial(m + n)) / ((factorial((m + n) - (n - a))) * factorial(n - a))
        Anm = Anm + factorial(n+m)* (Zeta)/factorial(a) * (-x)**a
    return Anm
#Spherical Ylm function uses the Associated Legendre function
def SphericalYlm (l,m,theta,phi):
    SphericalYlm = (-1**m)*np.sqrt(((2*l+1)/(4*np.pi))*((factorial(l-abs(m)))/(factorial(l+abs(m)))))\
                   *AssociatedLegendre(l,m,np.cos(theta))*np.exp(1j*m*phi)
    return SphericalYlm
def Y(l,m,theta, phi):
    if m < 0:
        Y = np.sqrt(2)*(-1)**m *np.imag(SphericalYlm(l,abs(m),theta,phi))
    elif m==0:
        Y = (-1**m)*np.sqrt(((2*l+1)/(4*np.pi))*((factorial(1-abs(m)))/(factorial(1+abs(m)))))\
                   *AssociatedLegendre(l,m,np.cos(theta))
    else:
        Y = np.sqrt(2)*(-1)**m *np.real(SphericalYlm(l,abs(m),theta,phi))

    return Y
def R(n,l,r):
    a=1
    R = (np.sqrt((2/(a*n))**3 * factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(a*n))*(2*r/(a*n))**l \
         /factorial(n-l-1+2*l+1)*AssociatedLaguerre(n - l - 1, 2 * l + 1, 2 * r / (a * n)))
    return R

# Wavefunction
def psi(n,l,m,r,theta,phi):
    psi = R(n,l,r)*Y(l,m,theta,phi)
    return psi
def psi_R(n,l,r):
    psi_R = r**2 * R(n,l,r)**2
    return psi_R
def psi_2(n,l,m,r,theta,phi):
    psi_2 = (r*R(n,l,r)*Y(l,m,theta,phi))**2
    return psi_2


def Plot_R(n,l):
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax= fig.add_subplot(1, 1, 1)
    newrange = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
    r = np.linspace(0, newrange, 100)
    plt.plot(r, psi_R(n, l, r,))
    plt.show()

def Plot_psi2plot1(n,l,m):
    size = np.round((2.5*n**2+1.5*n+1)/5)*5
    X2 = np.linspace(-size, size, 50)
    Y2 = np.linspace(-size, size, 50)
    [X2, Y2] = np.meshgrid(X2, Y2)
    rad = np.sqrt(X2**2+Y2**2)
    theta = 90
    phi = np.arctan2(Y2,X2)
    Z=psi_2(n,l,m,rad,theta,phi)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X2, Y2, Z)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    # ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()

def Plot_psi2plot2(n,l,m):
    size = np.round((2.5*n**2+1.5*n+1)/5)*5
    Y2 = np.linspace(-size, size, 50)
    Z2 = np.linspace(-size, size, 50)
    [Y2, Z2] = np.meshgrid(Y2, Z2)
    rad = np.sqrt(Y2**2+Z2**2)
    theta = np.arccos(Z2/rad)
    phi = 90
    X2=psi_2(n,l,m,rad,theta,phi)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(Y2, Z2, X2)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    # ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()
def Plot_psi2plot3(n,l,m):
    size = np.round((2.5*n**2+1.5*n+1)/5)*5
    X2 = np.linspace(-size, size, 50)
    Y2 = np.linspace(-size, size, 50)
    [X2, Y2] = np.meshgrid(X2, Y2)
    rad = np.sqrt(X2**2+Y2**2)
    theta = 90
    phi = np.arctan2(Y2,X2)
    Z=psi_2(n,l,m,rad,theta,phi)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X2, Y2, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()
def intensity_func(n,l,m,r,theta,phi):

    return (psi_2(n,l,m,abs(r),theta,phi))

def Plot_psi2plot4(n, l, m):
    size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
    Y2 = np.linspace(-size, size, 50)
    Z2 = np.linspace(-size, size, 50)
    [Y2, Z2] = np.meshgrid(Y2, Z2)
    rad = np.sqrt(Y2 ** 2 + Z2 ** 2)
    theta = np.arccos(Z2 / rad)
    phi = 90
    X2 = psi_2(n, l, m, rad, theta, phi)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y2, Z2, X2, cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()

def Ppsi (n,l,m):
    probabilitydensity = 1e-6
    size = np.round((2.5 * n ** 2 + 1.5 * n + 1) / 5) * 5
    border = 80
    accuracy = 150
    raster = np.linspace(-border, border, accuracy)
    [x, y, z] = np.meshgrid(raster, raster, raster)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    Wave = psi_2(n, l, m, r, theta, phi).real
    return Wave

def Plot_psi( n, l, m,):

  border = 100
  accuracy = 200 #Higher number smoother surface, but longer processign time.
  raster = np.linspace(-border, border, accuracy)
  [x, y, z] = np.meshgrid(raster, raster, raster)
  r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
  theta = np.arccos(z / r)
  phi = np.arctan2(y, x)
  Wave3 = psi_2(n, l, m, r, theta, phi)
  mlab.figure(1, fgcolor=(1, 1, 1))
  # We create a scalar field with the module of Phi as the scalar
  src = mlab.pipeline.scalar_field(Wave3)
  src.image_data.point_data.add_array(np.sign(psi(n,l,m,r,theta,phi)).T.ravel())
  src.image_data.point_data.get_array(1).name = 'phase'
  # Make sure that the dataset is up to date with the different arrays:
  src.update()
  # We select the 'scalar' attribute, ie the norm of Phi
  src2 = mlab.pipeline.set_active_attribute(src,
                                            point_scalars='scalar')
  # Cut isosurfaces of the norm
  contour = mlab.pipeline.contour(src2)
  contour.filter.contours=[0.0015,]
  # Now we select the 'angle' attribute, ie the phase of Phi
  contour2 = mlab.pipeline.set_active_attribute(contour,
                                                point_scalars='phase')
  # And we display the surface. The colormap is the current attribute: the phase.
  mlab.pipeline.surface(contour2, colormap='plasma',opacity=0.5)
  mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)
  mlab.show()




Plot_psi(n,l,m)
Plot_R(n,l)
#Plot_psi2plot1(n,l,m)
#Plot_psi2plot2(n,l,m)
#Plot_psi2plot3(n,l,m)
#Plot_psi2plot4(n,l,m)



#def hybridsp3a(n,l,m,r,theta,phi,n2,l2,m2):
   # hybrid1 = 0.25*psi(n2,l2,m2,r,theta,phi)+0.25*psi(n,l,m,r,theta,phi)+0.25*psi(n,l,m+1,r,theta,phi)+0.25*psi(n,l,m-1,r,theta,phi)
#    hybrid1 = (1/np.sqrt(4))*psi(n2,l2,m2,r,theta,phi) + (np.sqrt(3)/2)*psi(n,l,m,r,theta,phi)
#    return hybrid1
#def hybridsp3b(n,l,m,r,theta,phi,n2,l2,m2):
#    hybrid2 = 0.25*psi(n2,l2,m2,r,theta,phi)+0.25*psi(n,l,m,r,theta,phi)-0.25*psi(n,l,m+1,r,theta,phi)-0.25*psi(n,l,m-1,r,theta,phi)
#    return hybrid2
#def hybridsp3c(n,l,m,r,theta,phi,n2,l2,m2):
#    hybrid3 = 0.25*psi(n2,l2,m2,r,theta,phi)-0.25*psi(n,l,m,r,theta,phi)+0.25*psi(n,l,m+1,r,theta,phi)-0.25*psi(n,l,m-1,r,theta,phi)
#    return hybrid3
#def hybridsp3d(n,l,m,r,theta,phi,n2,l2,m2):
#    hybrid4 = 0.25*psi(n2,l2,m2,r,theta,phi)-0.25*psi(n,l,m,r,theta,phi)-0.25*psi(n,l,m+1,r,theta,phi)-0.25*psi(n,l,m-1,r,theta,phi)
#   return hybrid4


#def Plot_hybrid1(n, l, m, n2, l2, m2):
#    border = 100
#    accuracy = 200  # Higher number smoother surface, but longer processign time.
#    raster = np.linspace(-border, border, accuracy)
#    [x, y, z] = np.meshgrid(raster, raster, raster)
#    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#    theta = np.arccos(z / r)
#    phi = np.arctan2(y, x)
#    Wave1 = hybridsp3a(n, l, m, r, theta, phi, n2, l2, m2)
#    mlab.figure(1, fgcolor=(1, 1, 1))
#    # We create a scalar field with the module of Phi as the scalar
#    src = mlab.pipeline.scalar_field(Wave1)
#
#    src.image_data.point_data.add_array(np.sign(psi(n, l, m, r, theta, phi)).T.ravel())
#
#    src.image_data.point_data.get_array(1).name = 'phase'
#    # Make sure that the dataset is up to date with the different arrays:
#    src.update()
#
#    # We select the 'scalar' attribute, ie the norm of Phi
#    src2 = mlab.pipeline.set_active_attribute(src,
#                                              point_scalars='scalar')
#
#    # Cut isosurfaces of the norm
#    contour = mlab.pipeline.contour(src2)
#    contour.filter.contours = [0.002, ]
#
#    # Now we select the 'angle' attribute, ie the phase of Phi
#    contour2 = mlab.pipeline.set_active_attribute(contour,
#                                                  point_scalars='phase')
#
#    # And we display the surface. The colormap is the current attribute: the phase.
#    mlab.pipeline.surface(contour2, colormap='plasma', opacity=0.5)
#
#    mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)
#
#    mlab.show()
#def Plot_hybridsp3( n, l, m,n2,l2,m2):
#
#    border = 100
#    accuracy = 200 #Higher number smoother surface, but longer processign time.
#    raster = np.linspace(-border, border, accuracy)
#    [x, y, z] = np.meshgrid(raster, raster, raster)
#    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#    theta = np.arccos(z / r)
#    phi = np.arctan2(y, x)
#    Wave1 = hybridsp3a(n,l,m,r,theta,phi,n2,l2,m2)
#    mlab.figure(1, fgcolor=(1, 1, 1))
#    # We create a scalar field with the module of Phi as the scalar
#    src = mlab.pipeline.scalar_field(Wave1)
#
#    # Make sure that the dataset is up to date with the different arrays:
#    src.update()
#
#    # Cut isosurfaces of the norm
#    contour = mlab.pipeline.contour(src)
#    contour.filter.contours=[0.01,]
#
#
#    # And we display the surface. The colormap is the current attribute: the phase.
#    mlab.pipeline.surface(contour, colormap='plasma',opacity=0.9)
##222222222222222222222222222222
#    Wave2 = hybridsp3b(n,l,m,r,theta,phi,n2,l2,m2)
#
#    # We create a scalar field with the module of Phi as the scalar
#    src2 = mlab.pipeline.scalar_field(Wave2)
#
#    # Make sure that the dataset is up to date with the different arrays:
#    src2.update()
#
#    # Cut isosurfaces of the norm
#    contour = mlab.pipeline.contour(src2)
#    contour.filter.contours=[0.01,]
#
#
#    # And we display the surface. The colormap is the current attribute: the phase.
#    mlab.pipeline.surface(contour, colormap='autumn',opacity=0.9)
#    mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)
#
#    # 33333333333333333333333333333333333333333
#    Wave3 = hybridsp3c(n, l, m, r, theta, phi, n2, l2, m2)
#
#    # We create a scalar field with the module of Phi as the scalar
#    src3 = mlab.pipeline.scalar_field(Wave3)
#
#    # Make sure that the dataset is up to date with the different arrays:
#    src3.update()
#
#    # Cut isosurfaces of the norm
#    contour = mlab.pipeline.contour(src3)
#    contour.filter.contours = [0.01, ]
#
#    # And we display the surface. The colormap is the current attribute: the phase.
#    mlab.pipeline.surface(contour, colormap='summer', opacity=0.9)
#    mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)
#
#    # 4444444444444444444444444444444444444444444444
#    Wave4 = hybridsp3d(n, l, m, r, theta, phi, n2, l2, m2)
#
#    # We create a scalar field with the module of Phi as the scalar
#    src4 = mlab.pipeline.scalar_field(Wave4)
#
#    # Make sure that the dataset is up to date with the different arrays:
#    src4.update()
#
#    # Cut isosurfaces of the norm
#    contour = mlab.pipeline.contour(src4)
#    contour.filter.contours = [0.01, ]
#
#    # And we display the surface. The colormap is the current attribute: the phase.
#    mlab.pipeline.surface(contour, colormap='spring')
#    mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)
#    mlab.show()
#Plot_hybrid1( n, l, m,n2,l2,m2)
#Plot_hybridsp3( n, l, m,n2,l2,m2)