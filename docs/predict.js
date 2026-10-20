// Evaluates the exported LightGBM trees. x = [bedrooms, accommodates, minimum_nights, neighbourhood code or NaN].
function evalTree(node, x) {
  while (node.v === undefined) {
    let v = x[node.f];
    if (Number.isNaN(v)) {
      if (node.n) { node = node.d ? node.l : node.r; continue; }  // trained with missing values: follow default branch
      v = 0;                                                        // no missing in training: LightGBM treats NaN as 0
    }
    node = v <= node.t ? node.l : node.r;
  }
  return node.v;
}

// log-price is the sum of leaf values over all trees
function predictPrice(model, x) {
  let s = 0;
  for (const t of model.trees) s += evalTree(t, x);
  return Math.exp(s);
}

if (typeof module !== 'undefined') module.exports = { predictPrice };
