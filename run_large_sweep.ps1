$taus = @(0.45, 0.50, 0.55)
$ks = @(7, 9)
$closings = @(1, 2)
$pns = @(4, 6)
$rns = @(8, 12)

$base = "data\v2_sweep_big"
$i = 0

foreach ($tau in $taus) {
  foreach ($k in $ks) {
    foreach ($cl in $closings) {
      foreach ($pn in $pns) {
        foreach ($rn in $rns) {

          $out = "$base\run_$($i.ToString("000"))"

          python .\synthetic_trabecular_v2_voting3d.py `
            --outdir $out `
            --size 256 `
            --n-volumes 40 `
            --tau $tau `
            --k $k `
            --closing-iters $cl `
            --pn $pn `
            --rn $rn `
            --export-2d `
            --export-2d-mode mip

          $i++
        }
      }
    }
  }
}
