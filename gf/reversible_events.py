"""
Reversible events (e.g., migration and recombination)
do not yield simpler sampling configurations, contrary to non-reversible events (coalescence, population divergence).
Accordingly, for reversible events coupled equations emerge, the solution involving matrix inversion and higher-order
polynomials, turning infeasible for large n (Hobolth et al., 2011).
"""

"""
Using Taylor-series expansion the full GF can be decomposed into histories with 1, 2, ..., n events (Lohse
et al., 2011).

Basics for recombination (assume 2 haplotypes, and biallelic sites, panm.):
	loci		sample config.			possible rec. events	
	2			{{A1.B1},{A2.B2}}		1 * 2 = 2
	3			{{A1.B1.C1},...}		2 * 2 = 4
	4			{{A1.B1.C1.D1,...}		3 * 2 = 6
										(loci-1) * haplotypes, etc.										
"""

def give_psi(sample_configuration, n, events_left):
    # build the recursion and reduce
    pass

def get_error_of_approximation():
    pass
