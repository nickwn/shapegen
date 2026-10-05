"""Quad-mesh port, on a simplified track: cones come only from the
mesh-free spectral cross field, never from mesh curvature/
umbilics (not built and not planned on this track). End to end:
ρ̂ -> a quad mesh in ℝ³.

- :mod:`shapegen.quad.field` — the connection operator + Ginzburg-Landau
  anneal that produces the cones (no curvature-alignment blend term).
- :mod:`shapegen.quad.refine` — optional: locally optimise the *positions*
  of that (already correct) cone set against ``field.area_distortion``. Not
  the rejected "make cones emerge from a relaxation" — see its docstring.
- :mod:`shapegen.quad.cones` — pole choice + stereographic projection of
  those cones into a flat chart (minus the umbilic-splitting it did
  before projecting — the GL cones already carry the right ±¼ index).
- :mod:`shapegen.quad.periods` — the period solve: perturbs the chart
  within a small budget so every loop's developing-map period lands on the
  integer/half-integer lattice, which is what lets a square grid close up
  without seams. Reproduces the P0 corpus's real solved reference
  (``cones_solved.npy``/``cell.npy``) to 1e-8.
- :mod:`shapegen.quad.grid` — trace the developing map's grid over the
  whole sphere in two charts and assemble it from a rotation system into a
  **closed** quad mesh. On the P0 corpus: Euler characteristic 2, coverage
  1.000000000, no irregular vertex that is not a cone. Ground-truthed
  against the cube, whose flat structure has exactly these 8 cones and whose
  quad mesh at cell ``edge/m`` is known exactly.
- :mod:`shapegen.quad.lift` — lift the traced grid to ℝ³ from the spectral
  spinor, solved in spherical harmonics (no mesh, no edge graph).
- :mod:`shapegen.quad.pipeline` — the whole chain in one call,
  ``quad_mesh(rho_hat, L)``; ``scripts/quad_demo.py`` renders it.
"""
