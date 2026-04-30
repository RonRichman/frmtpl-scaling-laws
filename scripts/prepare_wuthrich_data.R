source_url <- "https://people.math.ethz.ch/~wueth/Lecture/freMTPL2freq.rda"
args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[[1]] else "data/FRMTPL.csv"
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

tmp_path <- tempfile(fileext = ".rda")
download.file(source_url, tmp_path, mode = "wb")
load(tmp_path)

if (!exists("freMTPL2freq")) {
  stop("Expected object `freMTPL2freq` in downloaded RDA.")
}

dat <- freMTPL2freq

RNGversion("3.5.0")
set.seed(500)
ll <- sample(c(1:nrow(dat)), round(0.9 * nrow(dat)), replace = FALSE)

dat$set <- "test"
dat$set[ll] <- "train"

set.seed(2024)
dat$sample_unif <- runif(nrow(dat))

write.csv(dat, out_path, row.names = FALSE)

cat("Wrote", nrow(dat), "rows to", out_path, "\n")
cat("train rows:", sum(dat$set == "train"), "\n")
cat("test rows:", sum(dat$set == "test"), "\n")
cat("train claims:", sum(dat$ClaimNb[dat$set == "train"]), "\n")
cat("test claims:", sum(dat$ClaimNb[dat$set == "test"]), "\n")
