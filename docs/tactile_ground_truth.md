# MuJoCo fingertip contact force output

DexJoCo exposes one force modality: `tactile_gt`. Its net-force fields reduce
MuJoCo contact constraints returned by `mujoco.mj_contactForce`; its spatial
field distributes those same contact forces over a deterministic 16-taxel
layout derived from the fingertip collision geometry.

## Pipeline

```text
mj_step at 500 Hz
  -> mj_contactForce for force-bearing fingertip contacts
  -> contact-frame sign and orientation handling
  -> torque shift from contact point to fingertip origin (r x F)
  -> fingertip body-local force and torque
  -> normalized Gaussian projection onto 16 taxels per fingertip
  -> 10 physics samples reduced into one 20 ms control observation
  -> observation["tactile_gt"]
  -> tactile_gt.* and next_tactile_gt.* in Zarr
```

The extractor and projector are read-only. They do not add geoms, constraints,
actuators, or forces, and therefore do not change the MuJoCo trajectory.

## Per-frame schema

Within each hand, fingertip order is `index, middle, ring, thumb`. Single-hand
tasks use `H=1`; bimanual tasks use `H=2` with hand order `right, left`.

| Field | Shape | Dtype | Unit | Control-window reduction |
| --- | --- | --- | --- | --- |
| `fingertip_wrench_local` | `(H, 4, 6)` | `float32` | N, N*m | Mean over all physics substeps |
| `taxel_force_local` | `(H, 4, 16, 3)` | `float32` | N | Mean of per-substep Gaussian projections |
| `fingertip_normal` | `(H, 4)` | `float32` | N | Mean summed normal load |
| `fingertip_force_peak` | `(H, 4)` | `float32` | N | Maximum summed normal load in one substep |
| `fingertip_impulse_local` | `(H, 4, 3)` | `float32` | N*s | Sum of local force times `physics_dt` |
| `contact_fraction` | `(H, 4)` | `float32` | 1 | Fraction of substeps above the contact threshold |
| `sim_time` | `(1,)` | `float64` | s | MuJoCo time at interval end |

The wrench channel order is `[Fx, Fy, Fz, Tx, Ty, Tz]`, expressed in each
fingertip body frame about that body's origin. Reset observations contain zero
interval statistics.

## Spatial taxel projection

Each fingertip uses a `4 axial rows x 4 circumferential columns` layout on its
collision capsule. Taxel centers and orthonormal axes include the compiled
MuJoCo geom position and quaternion. The taxel-local channels are
`[tangent_u, tangent_v, normal_inward]`.

The runtime fixes IDs `0..15` when the environment is initialized, using
`taxel_id = axial_row * 4 + circumferential_column`. MuJoCo does not produce a
taxel ID. It produces the contacting geom IDs and a continuous 3-D contact
position; the projector uses that position to compute the 16 weights.

For every force-bearing contact at position `p`, the projector computes one
weight for every taxel center `c_i`:

```text
w_i = exp(-||p - c_i||^2 / (2 sigma^2))
alpha_i = w_i / sum_j(w_j)
```

The default `sigma` is the median nearest-neighbour distance of the taxel
layout on that fingertip. The normalized weights distribute the source
contact force without changing its magnitude: after rotating every taxel-local
force back into the fingertip body frame, the sum reconstructs the original
MuJoCo contact force up to floating-point roundoff. The projection preserves
force; the original `fingertip_wrench_local` remains the authoritative output
for torque about the fingertip origin.

Projection is performed after every 2 ms physics step. Multiple contacts add
linearly within a substep, and all ten substeps—including zero-contact
substeps—participate in the 20 ms arithmetic mean. The mapping is fully
deterministic and uses only the MuJoCo model geometry, contact positions, and
contact forces.

## MuJoCo force handling

`mj_contactForce` returns a contact-frame wrench applied to `geom2`; the
equal-and-opposite wrench applies to `geom1`. The implementation resolves that
sign, rotates the wrench to the world frame, shifts torque to the fingertip
origin, and then rotates it into the fingertip body frame.

Contacts whose `efc_address < 0` are proximity-only and are excluded. The
extractor maps collision geoms through `model.geom_bodyid`, so it does not
depend on optional geom names.

Extraction runs immediately after every `mj_step`. A `mj_forward` must not be
inserted between the physics step and extraction because it would recompute
contact forces at a different state and corrupt peak/impulse accounting.

## Action alignment

Each Zarr row stores a complete transition:

```text
tactile_gt[t]       : force observation before action[t]
action[t]           : action executed during the row
next_tactile_gt[t]  : force observation after action[t]
```

The final action response is therefore retained instead of being dropped at
episode termination. `tactile_gt.sim_time` is the authoritative simulation
clock used to validate timestamps and `control_dt`.

## Zarr merge behavior

Raw episode arrays have a leading time dimension, for example
`fingertip_wrench_local` is `(T, H, 4, 6)` and `taxel_force_local` is
`(T, H, 4, 16, 3)`. New recordings use schema `dexjoco.tactile_gt.v2`. The
converter preserves rank and dtype, requires current and next streams to be
paired, and normalizes datasets to the canonical `[right, left]` hand axis.

Legacy complete v1 datasets remain readable when converted separately. A v1
dataset and a v2 dataset cannot be merged because the absent historical taxel
distribution is unknown and must not be represented as zero contact.

Merged datasets also contain:

- `sensor_present`: distinguishes an episode with no force stream from valid
  zero-contact force data.
- `hand_valid`: distinguishes available hands from zero padding.

Retired `tactile.*` streams are not part of the output contract and
must never be written into a new merged dataset.

## Verification

The test suite checks:

- `geom1/geom2` force sign against MuJoCo constraint force;
- rotated fingertip frames and `r x F` moment arms;
- zero output for proximity-only contacts;
- per-substep peak, impulse, mean, and contact fraction;
- 4-by-4 taxel geometry, normalized Gaussian weights, determinism, and force
  conservation;
- single-hand and bimanual taxel shapes and ten-substep temporal averaging;
- reset/step lifecycle and exact substep count in every integrated task;
- current/next Zarr alignment;
- single-hand, bimanual, and missing-stream merge behavior.

Run the main and converter suites from their respective package directories:

```bash
PYTHONPATH="$PWD:$PWD/../dexjoco-data-converter/src" \
python -m unittest discover -s tests -p 'test_*.py'

PYTHONPATH="$PWD/src" \
python -m unittest discover -s tests -p 'test_*.py'
```
