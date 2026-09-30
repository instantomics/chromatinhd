# ChromatinHD

This reference wraps ChromatinHD-*diff* from
[ChromatinHD](https://github.com/DeplanckeLab/ChromatinHD) 0.4.3 for the Iomix
`differential_accessibility` task. Upstream models the Tn5 insertions of each
cell in each region as a Poisson count scaled by library size. Their positions
follow a multiscale piecewise-constant density from 5 kb down to 25 bp. Every
cluster has its own region-level and positional deviations from a shared
baseline, and a normal prior shrinks the positional deviations. The model is
fitted by MAP with sparse Adam. The wrapper uses upstream's `binary.Model`,
`Shared` encoder, and trainer. Its one material change is how cluster
deviations are parameterized; see *Design sharing*.

## Wrapping choices

- **Clusters are design queries.** Each supplied query becomes one
  ChromatinHD cluster, and fit cells join clusters through the task's
  nearest-query assignment.
- **Design sharing.** Upstream fits each cluster's deviations freely. The
  wrapper instead fits them as linear effects of a query design matrix, at
  every positional resolution and for the region totals. This follows
  upstream's low-rank encoder, which expresses deviations through cluster
  covariates, but uses task design columns instead of transcriptome
  components. The design contains cell type, cell type times each other
  categorical design column, and cell type times a three-knot piecewise-linear
  basis over each continuous column. Columns fixed within a cell type add
  nothing. Upstream's normal prior still applies to every cluster's positional
  deviation. A query without fit cells is predicted from its design effects.
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
- **Training.** One model is fitted to all fit cells with upstream's
  optimizer defaults: 30 epochs, learning rate 0.01, and minibatches of 250
  cells by 100 regions. No cells are split off for validation, which matches
  upstream's default of no early stopping. The prior scale on deviations is
  0.25 rather than upstream's 1.5. The seed is 1729, and training runs on four
  CPU threads in the task's adapted route.

Upstream publishes only a Cython sdist.
[`scripts/build_chromatinhd_wheel.sh`](scripts/build_chromatinhd_wheel.sh)
builds it into the Linux wheel published as this repository's
`chromatinhd-wheel-v0.4.3` release asset. The script checks the sdist's
SHA-256, uses pinned build tools, and requires GCC 14.3. The candidate lock
pins that wheel. Upstream imports `polyptich` without declaring it. The
candidate session gets it from the Iomix framework, which depends on it.

## Scientific notes

Synthetic worlds spread about 600 cells over 144 design queries. There,
upstream's free per-cluster deviations overfit: baseline and differential
recovery stayed below the additive pseudobulk anchor for 1 to 30 training
epochs. Design sharing makes differential recovery positive. With upstream's
prior scale, baseline recovery still stayed below the anchor, and longer
training did not help. The prior scale was chosen among 1.5, 0.5, and 0.25 and
the basis among two, three, and five knots, using only the public training
worlds and observed training dataset. At the selected setting, both recovery
components exceed the anchor on every synthetic training world. On the observed
JVG28 zonation dataset, where each query has tens to hundreds of cells,
held-out prediction is well above the anchor and essentially unchanged from
free per-cluster fitting. Differential structure is therefore limited to
additive effects within cell types and to piecewise-linear continuous
responses.

The wrapper uses no held-out cells, truth, validation outcome, or RNA. It does
not tune from validation or retry. Its rates are the model's point estimates,
without posterior uncertainty or calibrated differential calls.

There is no `differential_coaccessibility` integration. ChromatinHD-*diff*
treats the cells of one cluster as independent draws, so, unchanged, it implies
a co-accessibility ratio of one everywhere. That answer is the task's
no-coupling zero point. The long-range interactions described for ChromatinHD
come from ChromatinHD-*pred* co-predictivity: the non-additive effect of jointly
censoring two windows on predicted gene expression. Computing it needs paired
RNA, which that task does not give candidates, and it measures joint prediction
of expression rather than within-cell co-opening.
