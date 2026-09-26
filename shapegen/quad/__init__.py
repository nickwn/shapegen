"""Quad-mesh port (PLAN.md), on a simplified track: cones come only from the
mesh-free spectral cross field (§P8.3/§P8.4), never from mesh curvature/
umbilics (§P2/§P3, not built and not planned on this track). End to end:
ρ̂ -> a quad mesh in ℝ³.

- :mod:`shapegen.quad.field` — the connection operator + Ginzburg-Landau
  anneal that produces the cones (§P8.3/§P8.4, no §P8.6 blend term).
- :mod:`shapegen.quad.refine` — optional: locally optimise the *positions*
  of that (already correct) cone set against ``field.area_distortion``. Not
  §P8.7's rejected "make cones emerge from a relaxation" — see its docstring.
- :mod:`shapegen.quad.cones` — pole choice + stereographic projection of
  those cones into a flat chart (§P4, minus the umbilic-splitting it did
  before projecting — the GL cones already carry the right ±¼ index).
- :mod:`shapegen.quad.periods` — the period solve (§P5): perturbs the chart
  within a small budget so every loop's developing-map period lands on the
  integer/half-integer lattice, which is what lets a square grid close up
  without seams. Reproduces the P0 corpus's real solved reference
  (``cones_solved.npy``/``cell.npy``) to 1e-8.
- :mod:`shapegen.quad.grid` — §P6: trace the developing map's grid over the
  whole sphere in two charts and assemble it from a rotation system into a
  **closed** quad mesh. On the P0 corpus: Euler characteristic 2, coverage
  1.000000000, no irregular vertex that is not a cone. Ground-truthed
  against the cube, whose flat structure has exactly these 8 cones and whose
  quad mesh at cell ``edge/m`` is known exactly.
- :mod:`shapegen.quad.extract` — the bundle's original single-chart
  extraction, **superseded by** :mod:`~shapegen.quad.grid` and kept only
  because it reproduces the P0 corpus's saved
  ``q2_pos.npy``/``q2_F.npy``/``q2_adj.npy`` exactly. It tops out at 91%
  coverage with a badly broken valence histogram; see its docstring.
- :mod:`shapegen.quad.lift` — §P7: lift the traced grid to ℝ³. Route 2
  (the default) integrates the §P9 spinor along the traced edges, no mesh;
  route 1 interpolates :func:`shapegen.codec.decode` and is the reference.

``scripts/quad_demo.py`` runs this whole chain on real face ρ fields and
renders the extracted mesh. The public ``quad_mesh() -> QuadResult`` surface
described in PLAN.md §4 (bundling all of this behind one call) is not built
yet, but every stage it would call already exists and is tested.
"""
