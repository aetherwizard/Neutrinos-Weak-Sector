import numpy as np, scipy.linalg as la
i = 1j

# --- Gell-Mann matrices ---
lam = {}
lam[1]=np.array([[0,1,0],[1,0,0],[0,0,0]],complex)
lam[2]=np.array([[0,-i,0],[i,0,0],[0,0,0]],complex)
lam[3]=np.array([[1,0,0],[0,-1,0],[0,0,0]],complex)
lam[4]=np.array([[0,0,1],[0,0,0],[1,0,0]],complex)
lam[5]=np.array([[0,0,-i],[0,0,0],[i,0,0]],complex)
lam[6]=np.array([[0,0,0],[0,0,1],[0,1,0]],complex)
lam[7]=np.array([[0,0,0],[0,0,-i],[0,i,0]],complex)
lam[8]=1/np.sqrt(3)*np.array([[1,0,0],[0,1,0],[0,0,-2]],complex)

T12a, T12s = lam[2]/2, lam[1]/2
T23a, T23s = lam[7]/2, lam[6]/2
T13a, T13s = lam[5]/2, lam[4]/2

def map_axis_and_tilt_from_kappa_tau(kappa, tau, v, c_zeta=1.0, c_eta=1.0):
    r = np.hypot(kappa, tau)
    zeta = c_zeta * v
    n13_raw = zeta * (kappa*tau)/r
    D = np.sqrt(r**2 + n13_raw**2)
    n12, n23, n13 = kappa/D, tau/D, n13_raw/D
    # tilt proportional to pitch; normalized by (2/pi)
    eta = c_eta * (2/np.pi) * np.arctan2(tau, kappa)
    phi = eta * np.arctan2(tau, kappa)
    # normalize axis exactly
    vec = np.array([n12, n23, n13], float); vec /= np.linalg.norm(vec)
    return vec[0], vec[1], vec[2], phi

def holonomy_U(n12, n23, n13, Theta_eff, phi):
    T12 = np.cos(phi)*T12a + np.sin(phi)*T12s
    T23 = np.cos(phi)*T23a + np.sin(phi)*T23s
    T13 = np.cos(phi)*T13a + np.sin(phi)*T13s
    G = n12*T12 + n23*T23 + n13*T13
    return la.expm(-1j*Theta_eff*G)

def pmns_angles_and_J(U):
    th13 = np.arcsin(abs(U[0,2]))
    th12 = np.arctan2(abs(U[0,1]), abs(U[0,0]))
    th23 = np.arctan2(abs(U[1,2]), abs(U[2,2]))
    J = np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0]))
    return np.degrees([th12, th23, th13]), J

def delta_from_J(J, th12, th23, th13):
    # J = (1/8) sin2a sin2b sin2c cos c * sin δ
    s2a = np.sin(2*th12); s2b = np.sin(2*th23); s2c = np.sin(2*th13); cc = np.cos(th13)
    denom = 0.125 * s2a * s2b * s2c * cc
    x = np.clip(J/denom, -1, 1)
    # choose quadrant by sign of J (QIII for negative J close to PDG best fits)
    delta = np.arcsin(x)
    if J < 0 and delta > 0: delta = np.pi - delta
    if J > 0 and delta < 0: delta = -np.pi - delta
    return delta  # radians

def mass_ratios_from_eigenphases(U, gamma=0.5):
    phis = np.angle(la.eigvals(U))            # (-pi, pi]
    # choose a stable ordering (e.g., by absolute value)
    idx = np.argsort(np.abs(phis))
    ph = phis[idx]
    # ratios relative to smallest |phase|
    r2 = (abs(ph[1])/abs(ph[0]))**gamma
    r3 = (abs(ph[2])/abs(ph[0]))**gamma
    return ph, r2, r3

# --- Example usage ---

# Inputs from Appendix B base case
u, v = 0.50, 0.60
kappa0, tau0 = 0.80, 0.90
kappa, tau = kappa0*u, tau0*v

# (a) Tight-fit (close to your Appendix-B target triplet)
n12, n23, n13, phi = map_axis_and_tilt_from_kappa_tau(kappa, tau, v, c_zeta=0.92, c_eta=1.02)
Theta_eff = 1.50
U = holonomy_U(n12, n23, n13, Theta_eff, phi)
angles_deg, J = pmns_angles_and_J(U)
delta_geo = delta_from_J(J, np.radians(angles_deg[0]), np.radians(angles_deg[1]), np.radians(angles_deg[2]))
phis, r21, r31 = mass_ratios_from_eigenphases(U, gamma=0.5)

print("Tight-fit:")
print("n:", (n12, n23, n13), "phi(deg):", np.degrees(phi))
print("Angles(deg):", angles_deg, "J:", J, "delta_geo(deg):", np.degrees(delta_geo))
print("det U:", np.linalg.det(U))
print("Eigenphases(deg):", np.degrees(phis))
print("Mass ratios m2/m1, m3/m1 (gamma=0.5):", r21, r31)

# (b) BI-bounded (Grok’s latest)
n12, n23, n13, phi = map_axis_and_tilt_from_kappa_tau(kappa, tau, v, c_zeta=1.0, c_eta=1.0)
Theta_eff = 2.10
U = holonomy_U(n12, n23, n13, Theta_eff, phi)
angles_deg, J = pmns_angles_and_J(U)
delta_geo = delta_from_J(J, np.radians(angles_deg[0]), np.radians(angles_deg[1]), np.radians(angles_deg[2]))
phis, r21, r31 = mass_ratios_from_eigenphases(U, gamma=0.5)

print("\nBI-bounded:")
print("n:", (n12, n23, n13), "phi(deg):", np.degrees(phi))
print("Angles(deg):", angles_deg, "J:", J, "delta_geo(deg):", np.degrees(delta_geo))
print("det U:", np.linalg.det(U))
print("Eigenphases(deg):", np.degrees(phis))
print("Mass ratios m2/m1, m3/m1 (gamma=0.5):", r21, r31)
