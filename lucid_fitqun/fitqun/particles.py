"""Particle constants shared by the tuning generators.

Kept free of numpy (and of anything else) so the cluster fan-out, which runs
on the submit host rather than inside the container, can generate PhotonSim
macros without a scientific-Python stack.
"""
from __future__ import annotations

from lucid.pdg import PDG_MASS_MEV

# fiTQun's particle hypotheses, by PDG code.
PDG_NAMES = {11: "e-", 13: "mu-", 211: "pi+"}

# Rest masses (MeV) for the momentum -> kinetic-energy conversion.
PDG_MASSES = {pdg: PDG_MASS_MEV[pdg] for pdg in PDG_NAMES}
