# freMTPL2 Data Attribution and License Notes

This directory includes a freMTPL2-style CSV for the public scaling-law demo.
The bundled CSV is generated from Mario Wüthrich's corrected frequency RDA:

https://people.math.ethz.ch/~wueth/Lecture/freMTPL2freq.rda

The learning/test split follows Wüthrich and Merz, Listing 5.2:

```r
RNGversion("3.5.0")
set.seed(500)
ll <- sample(c(1:nrow(dat)), round(0.9*nrow(dat)), replace = FALSE)
learn <- dat[ll,]
test <- dat[-ll,]
```

This gives 610,206 learning rows and 67,801 test rows in the bundled CSV.

Primary references:

- Wüthrich, M. V. and Merz, M. (2023). *Statistical Foundations of Actuarial
  Learning and its Applications*. Appendix B: Data and Examples.
  https://link.springer.com/book/10.1007/978-3-031-12409-9
- CASdatasets documentation: https://dutangc.github.io/CASdatasets/reference/freMTPL.html
- CASdatasets package metadata lists the package license as GPL (>= 2).

This companion package is distributed under GPL-2.0-or-later terms because the
public data source is GPL-aligned and the package may include a materialized CSV
derived from that source. Users are responsible for checking the current terms
of any data source or mirror they use.

The paper’s proprietary motor portfolio is not included here. This public dataset is used only to reproduce the methodology and teaching workflow at smaller scale.
