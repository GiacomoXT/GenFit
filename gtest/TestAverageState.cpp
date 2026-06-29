// Tests for genfit::tools::averageState, the numerical kernel of
// genfit::calcAverageState().
//
// calcAverageState used to build two ROOT TDecompChol objects to Cholesky-
// factor the covariances.  tools::averageState inlines that factorisation
// (dropping the per-call TDecompChol object) and reuses the optimized
// QR / transposedInvert.  The inlined Cholesky scales each row by a reciprocal
// (1/ujj) instead of dividing by ujj as ROOT does, so it is not bit-identical
// to the original; these tests assert that
//   * it agrees with the real ROOT TDecompChol code path to a tight relative
//     tolerance, and
//   * it reproduces the textbook inverse-variance weighted average,
// across a range of covariance dimensions, and that a non-positive-definite
// covariance is rejected.

#include <gtest/gtest.h>

#include <Tools.h>

#include <TDecompChol.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TVectorD.h>

#include <cmath>
#include <random>

namespace genfit {

namespace {

// The original calcAverageState() numerical body, verbatim, driven by the real
// ROOT TDecompChol.  Golden reference for the agreement test.
bool refROOT(const TVectorD& x1, const TMatrixDSym& C1,
             const TVectorD& x2, const TMatrixDSym& C2,
             TVectorD& avgState, TMatrixD& inv)
{
  TDecompChol d1(C1);
  bool success = d1.Decompose();
  TDecompChol d2(C2);
  success &= d2.Decompose();
  if (!success)
    return false;

  const int nRows = d1.GetU().GetNrows();
  TMatrixD S1inv, S2inv;
  tools::transposedInvert(d1.GetU(), S1inv);
  tools::transposedInvert(d2.GetU(), S2inv);

  TMatrixD A(2*nRows, nRows);
  TVectorD b(2*nRows);
  double *const bk = b.GetMatrixArray();
  double *const Ak = A.GetMatrixArray();
  const double* S1invk = S1inv.GetMatrixArray();
  const double* S2invk = S2inv.GetMatrixArray();
  for (int i = 0; i < nRows; ++i) {
    double sum1 = 0;
    double sum2 = 0;
    for (int j = 0; j <= i; ++j) {
      Ak[i*nRows + j] = S1invk[i*nRows + j];
      Ak[(i + nRows)*nRows + j] = S2invk[i*nRows + j];
      sum1 += S1invk[i*nRows + j]*x1.GetMatrixArray()[j];
      sum2 += S2invk[i*nRows + j]*x2.GetMatrixArray()[j];
    }
    bk[i] = sum1;
    bk[i + nRows] = sum2;
  }

  tools::QR(A, b);
  A.ResizeTo(nRows, nRows);

  tools::transposedInvert(A, inv);
  b.ResizeTo(nRows);
  for (int i = 0; i < nRows; ++i) {
    double sum = 0;
    for (int j = i; j < nRows; ++j)
      sum += inv.GetMatrixArray()[j*nRows+i] * b[j];
    b.GetMatrixArray()[i] = sum;
  }

  avgState.ResizeTo(nRows);
  avgState = b;
  return true;
}

// Random SPD covariance C = M^T M + (n+1) I and random state x.
void makeSPD(TMatrixDSym& C, TVectorD& x, int n, std::mt19937_64& rng)
{
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  TMatrixD M(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      M(i, j) = d(rng);
  C.ResizeTo(n, n);
  C = TMatrixDSym(TMatrixDSym::kAtA, M);
  for (int i = 0; i < n; ++i)
    C(i, i) += (n + 1.0);
  x.ResizeTo(n);
  for (int i = 0; i < n; ++i)
    x(i) = d(rng);
}

double maxAbs(const TMatrixD& M)
{
  double m = 0;
  const int N = M.GetNrows() * M.GetNcols();
  const double* p = M.GetMatrixArray();
  for (int i = 0; i < N; ++i) m = std::max(m, std::fabs(p[i]));
  return m;
}

} // anonymous namespace

// Agreement with the real ROOT TDecompChol code path, to a tight relative
// tolerance (the reciprocal-multiply Cholesky is not bit-identical to ROOT's
// division, but differs only by ~1 ulp per element).
TEST(AverageState, AgreesWithRootTDecompChol)
{
  std::mt19937_64 rng(20240629);
  const double tol = 1e-10;
  double worstState = 0.0, worstCov = 0.0;

  for (int n : {1, 2, 3, 4, 5, 6, 7, 10, 15, 25}) {
    for (int t = 0; t < 200; ++t) {
      TMatrixDSym C1, C2; TVectorD x1, x2;
      makeSPD(C1, x1, n, rng);
      makeSPD(C2, x2, n, rng);

      TVectorD sRef, sOpt;
      TMatrixD covRef, covOpt;
      ASSERT_TRUE(refROOT(x1, C1, x2, C2, sRef, covRef));
      ASSERT_TRUE(tools::averageState(x1, C1, x2, C2, sOpt, covOpt));

      double scaleS = 1e-300, devS = 0.0;
      for (int i = 0; i < n; ++i) scaleS = std::max(scaleS, std::fabs(sRef(i)));
      for (int i = 0; i < n; ++i) devS = std::max(devS, std::fabs(sOpt(i) - sRef(i)));
      worstState = std::max(worstState, devS / scaleS);

      double scaleC = std::max(1e-300, maxAbs(covRef));
      TMatrixD diff(covOpt); diff -= covRef;
      worstCov = std::max(worstCov, maxAbs(diff) / scaleC);
    }
  }

  EXPECT_LT(worstState, tol) << "worst relative state deviation vs ROOT";
  EXPECT_LT(worstCov,   tol) << "worst relative cov deviation vs ROOT";
}

// Reproduces the textbook inverse-variance weighted average, checked with
// ROOT's dense inversion:  avgCov == (C1^-1 + C2^-1)^-1  and
//                          avgState == avgCov (C1^-1 x1 + C2^-1 x2).
TEST(AverageState, MatchesTextbookAverage)
{
  std::mt19937_64 rng(987654321);
  const double tol = 1e-8;
  double worst = 0.0;

  for (int n : {1, 2, 3, 5, 6, 7, 12}) {
    for (int t = 0; t < 200; ++t) {
      TMatrixDSym C1, C2; TVectorD x1, x2;
      makeSPD(C1, x1, n, rng);
      makeSPD(C2, x2, n, rng);

      TVectorD sOpt; TMatrixD covFactor;
      ASSERT_TRUE(tools::averageState(x1, C1, x2, C2, sOpt, covFactor));

      TMatrixDSym C1inv(C1); C1inv.Invert();
      TMatrixDSym C2inv(C2); C2inv.Invert();
      TMatrixDSym sumInv(C1inv); sumInv += C2inv;
      TMatrixDSym avgCovExp(sumInv); avgCovExp.Invert();
      TVectorD stateExp = avgCovExp * ((C1inv * x1) + (C2inv * x2));

      TMatrixDSym avgCov(TMatrixDSym::kAtA, covFactor);

      double scaleC = 1e-300;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          scaleC = std::max(scaleC, std::fabs(avgCovExp(i, j)));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          worst = std::max(worst, std::fabs(avgCov(i, j) - avgCovExp(i, j)) / scaleC);

      double scaleS = 1e-300;
      for (int i = 0; i < n; ++i) scaleS = std::max(scaleS, std::fabs(stateExp(i)));
      for (int i = 0; i < n; ++i)
        worst = std::max(worst, std::fabs(sOpt(i) - stateExp(i)) / scaleS);
    }
  }

  EXPECT_LT(worst, tol) << "worst relative averaging residual";
}

// A non-positive-definite covariance must be rejected (return false).
TEST(AverageState, RejectsNonPositiveDefinite)
{
  std::mt19937_64 rng(13);
  const int n = 5;
  TMatrixDSym C1, C2; TVectorD x1, x2;
  makeSPD(C1, x1, n, rng);
  makeSPD(C2, x2, n, rng);
  C1(2, 2) = -1.0;   // break positive-definiteness

  TVectorD s; TMatrixD cov;
  EXPECT_FALSE(tools::averageState(x1, C1, x2, C2, s, cov));
}

} // namespace genfit
