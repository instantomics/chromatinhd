# ChromatinHD

This reference wraps ChromatinHD-*diff* from
[ChromatinHD](https://github.com/DeplanckeLab/ChromatinHD) 0.4.3 for the Iomix
`differential_accessibility` task. Upstream models the Tn5 insertions of each
cell in each region as a Poisson count scaled by library size. Their positions
follow a multiscale piecewise-constant density from 5 kb down to 25 bp. Every
cluster has its own region-level and positional deviations from a shared
baseline, and a normal prior shrinks the positional deviations. The model is
fitted by MAP with sparse Adam. The upstream `binary.Model`, its `Shared`
encoder, and its trainer are used unchanged.

## Wrapping choices

- **Clusters are design queries.** Each supplied query becomes one
  ChromatinHD cluster, and fit cells join clusters through the task's
  nearest-query assignment. Cell type, condition, nuisance setting, and
  continuous grid point therefore each get their own landscape. Apart from the
  shared baseline, ChromatinHD-*diff* has no structure for sharing across design
  factors or neighbouring continuous values.
- **Insertions and exposure.** Each task fragment row is one Tn5 insertion, so
  it is stored as a single-cut fragment. The task's declared exposure replaces
  upstream's fragment-count library size. The fixed region bias uses upstream's
  initialization formula in exposure units, so fitted rates are per unit
  exposure.
- **Regions and resolution.** Regions keep the genomic bin order on both
  strands. The common window is the widest region rounded up to 25 bp. Upstream
  resolutions that do not tile the window are dropped: 20 kb TSS windows use all
  seven levels, and 800 bp synthetic regions use 200 to 25 bp. A bin's rate is
  the region rate times the fitted density integrated over the bin.
- **Queries without fit cells.** A query with no fit cells uses the nearest
  represented continuous grid point that has the same categorical design. If no
  such point exists, it keeps ChromatinHD's prediction for an unobserved
  cluster, which is the shared baseline. This deterministic fallback is the
  wrapper's only addition to the upstream model.
- **Training.** One model is fitted to all fit cells with upstream's defaults:
  30 epochs, learning rate 0.01, and minibatches of 250 cells by 100 regions. No
  cells are split off for validation, which matches upstream's default of no
  early stopping. The seed is 1729, and training runs on four CPU threads in
  the task's adapted route.

Upstream publishes only a Cython sdist.
[`scripts/build_chromatinhd_wheel.sh`](scripts/build_chromatinhd_wheel.sh)
builds it into the Linux wheel published as this repository's
`chromatinhd-wheel-v0.4.3` release asset. The script checks the sdist's
SHA-256, uses pinned build tools, and requires GCC 14.3. The candidate lock
pins that wheel. Upstream imports `polyptich` without declaring it. The
candidate session gets it from the Iomix framework, which depends on it.

## Scientific notes

Public training evidence shows two regimes. On the observed JVG28 zonation
dataset, where each query has tens to hundreds of cells, ChromatinHD-*diff*
predicts held-out cells well above the pseudobulk anchor. On synthetic worlds,
which spread about 600 cells over 144 design queries, the independent per-query
clusters overfit. There, baseline and differential recovery stay below the
additive pseudobulk anchor for 1 to 30 training epochs. This is a property of
unshared cluster-wise modelling in sparse factorial designs, not of the
positional model itself.

The wrapper uses no held-out cells, truth, validation outcome, or RNA. It does
not tune from validation or retry. Its rates are the model's point estimates,
without posterior uncertainty or calibrated differential calls.
