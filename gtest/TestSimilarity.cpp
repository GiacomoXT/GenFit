// Tests guarding the optimization of genfit::KalmanFitterRefTrack::
// processTrackPoint (non-square-root path).
//
// The only numerical change in processTrackPoint is replacing ROOT's
// TMatrixDSym::Similarity with genfit::tools::similarity for the three
// similarity transforms it performs:
//   1. C_.Similarity(F)            (covariance prediction)
//   2. covSumInv_.Similarity(CHt)  (covariance update term)
//   3. Rinv_.Similarity(res_)      (chi2 increment, scalar/vector form)
//
// These tests therefore:
//   * SimilarityMatrix / SimilarityVector - pin tools::similarity against the
//     real ROOT TMatrixDSym::Similarity over the matrix shapes the fitter uses
//     (and beyond), to a tight relative tolerance;
//   * ProcessTrackPointKernel - replays the exact predict + update arithmetic
//     of processTrackPoint twice, once with ROOT's Similarity and once with
//     tools::similarity (everything else identical), and requires the updated
//     state, covariance and chi2 to agree to a tight tolerance. This is the
//     runnable stand-in for "processTrackPoint does not change behaviour".

#include <gtest/gtest.h>

#include <Tools.h>

#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TVectorD.h>

#include <cmath>
#include <random>

namespace genfit {

namespace {

const double kTol = 1e-10;

double maxAbs(const TMatrixDSym& M)
{
  double m = 0;
  const int N = M.GetNrows() * M.GetNcols();
  const double* p = M.GetMatrixArray();
  for (int i = 0; i < N; ++i) m = std::max(m, std::fabs(p[i]));
  return m;
}

double maxRelDiff(const TMatrixDSym& a, const TMatrixDSym& b)
{
  const double scale = std::max(1e-300, std::max(maxAbs(a), maxAbs(b)));
  double d = 0;
  for (int i = 0; i < a.GetNrows(); ++i)
    for (int j = 0; j < a.GetNcols(); ++j)
      d = std::max(d, std::fabs(a(i, j) - b(i, j)));
  return d / scale;
}

double maxRelDiff(const TVectorD& a, const TVectorD& b)
{
  double scale = 1e-300, d = 0;
  for (int i = 0; i < a.GetNrows(); ++i) scale = std::max(scale, std::fabs(a(i)));
  for (int i = 0; i < a.GetNrows(); ++i) d = std::max(d, std::fabs(a(i) - b(i)));
  return d / scale;
}

TMatrixD randMatrix(int r, int c, std::mt19937_64& rng)
{
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  TMatrixD M(r, c);
  for (int i = 0; i < r * c; ++i) M.GetMatrixArray()[i] = d(rng);
  return M;
}

// random SPD n x n = G G^T + n I
TMatrixDSym randSPD(int n, std::mt19937_64& rng)
{
  TMatrixD G = randMatrix(n, n, rng);
  TMatrixDSym S(TMatrixDSym::kAtA, G);   // G^T G
  for (int i = 0; i < n; ++i) S(i, i) += n;
  return S;
}

TVectorD randVector(int n, std::mt19937_64& rng)
{
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  TVectorD v(n);
  for (int i = 0; i < n; ++i) v(i) = d(rng);
  return v;
}

} // anonymous namespace


// tools::similarity(B, sym) == TMatrixDSym::Similarity(B), within tolerance,
// across square (the F transport case) and rectangular (the C*H^T case) shapes.
TEST(SimilarityMatrix, MatchesRoot)
{
  std::mt19937_64 rng(20240630);
  double worst = 0.0;

  // (M rows of B, N = state/sym dimension); covers M==N, M>N (C*H^T: dim x m),
  // M<N, plus a few larger generic shapes.
  for (int N = 1; N <= 6; ++N) {
    for (int M = 1; M <= 6; ++M) {
      for (int t = 0; t < 100; ++t) {
        TMatrixDSym sym = randSPD(N, rng);
        TMatrixD B = randMatrix(M, N, rng);

        TMatrixDSym ref(sym); ref.Similarity(B);     // ROOT
        TMatrixDSym opt(sym); tools::similarity(B, opt);

        ASSERT_EQ(opt.GetNrows(), M);
        ASSERT_EQ(opt.GetNcols(), M);
        worst = std::max(worst, maxRelDiff(ref, opt));
      }
    }
  }
  // a couple of larger shapes
  for (auto pr : {std::pair<int,int>{12, 8}, {8, 12}, {20, 20}}) {
    TMatrixDSym sym = randSPD(pr.second, rng);
    TMatrixD B = randMatrix(pr.first, pr.second, rng);
    TMatrixDSym ref(sym); ref.Similarity(B);
    TMatrixDSym opt(sym); tools::similarity(B, opt);
    worst = std::max(worst, maxRelDiff(ref, opt));
  }

  EXPECT_LT(worst, kTol) << "tools::similarity(B,sym) vs ROOT, worst rel diff = " << worst;
}

// tools::similarity(v, A) == A.Similarity(v), within tolerance.
TEST(SimilarityVector, MatchesRoot)
{
  std::mt19937_64 rng(424242);
  double worst = 0.0;
  for (int n = 1; n <= 8; ++n) {
    for (int t = 0; t < 200; ++t) {
      TMatrixDSym A = randSPD(n, rng);
      TVectorD v = randVector(n, rng);
      double ref = A.Similarity(v);
      double opt = tools::similarity(v, A);
      double scale = std::max(1e-300, std::fabs(ref));
      worst = std::max(worst, std::fabs(ref - opt) / scale);
    }
  }
  EXPECT_LT(worst, kTol) << "tools::similarity(v,A) vs ROOT, worst rel diff = " << worst;
}


// Replays processTrackPoint's predict + single-measurement update arithmetic,
// switching only the three Similarity calls between ROOT and tools, and
// requires the resulting (state, cov, chi2) to agree.
TEST(ProcessTrackPointKernel, MatchesRoot)
{
  std::mt19937_64 rng(7);
  double worstP = 0.0, worstC = 0.0, worstChi2 = 0.0;

  for (int dim : {5, 6}) {
    for (int m : {1, 2, dim}) {        // measurement dimension
      for (int t = 0; t < 200; ++t) {
        // prediction inputs
        TMatrixDSym Cprev = randSPD(dim, rng);
        TVectorD xprev = randVector(dim, rng);
        TMatrixD F = randMatrix(dim, dim, rng);
        TMatrixDSym N = randSPD(dim, rng);
        TVectorD delta = randVector(dim, rng);
        // measurement inputs
        TMatrixD H = randMatrix(m, dim, rng);
        TMatrixDSym V = randSPD(m, rng);
        TVectorD meas = randVector(m, rng);

        // useTools: switch ONLY the three Similarity calls.
        auto run = [&](bool useTools, TVectorD& pOut, TMatrixDSym& COut, double& chi2Out) {
          // ---- predict ----
          TVectorD p = F * xprev + delta;
          TMatrixDSym C(Cprev);
          if (useTools) tools::similarity(F, C); else C.Similarity(F);  // (1)
          C += N;

          // ---- update (single measurement) ----
          // S = H C H^T + V    (computed identically in both arms; HMHt is not
          // a TMatrixDSym::Similarity call in processTrackPoint and is unchanged)
          TMatrixD HC(H, TMatrixD::kMult, C);
          TMatrixDSym S(m);
          for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
              double s = 0;
              for (int k = 0; k < dim; ++k) s += HC(i, k) * H(j, k);
              S(i, j) = s + V(i, j);
            }
          TMatrixDSym Sinv(S);
          tools::invertMatrix(Sinv);

          TMatrixD CHt(C, TMatrixD::kMultTranspose, H);   // dim x m
          TVectorD res = meas - H * p;
          if (useTools)
            p += CHt * (Sinv * res);                       // reassociated Kalman gain
          else
            p += TMatrixD(CHt, TMatrixD::kMult, Sinv) * res;

          TMatrixDSym covTerm(Sinv);
          if (useTools) tools::similarity(CHt, covTerm); else covTerm.Similarity(CHt);  // (2)
          C -= covTerm;

          // ---- chi2 increment ----
          TVectorD resNew = meas - H * p;
          TMatrixD HC2(H, TMatrixD::kMult, C);
          TMatrixDSym Rinv(m);
          for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
              double s = 0;
              for (int k = 0; k < dim; ++k) s += HC2(i, k) * H(j, k);
              Rinv(i, j) = V(i, j) - s;          // V - H C H^T
            }
          tools::invertMatrix(Rinv);
          double chi2 = useTools ? tools::similarity(resNew, Rinv) : Rinv.Similarity(resNew); // (3)

          pOut.ResizeTo(p); pOut = p;
          COut.ResizeTo(C); COut = C;
          chi2Out = chi2;
        };

        TVectorD pRef, pOpt; TMatrixDSym CRef, COpt; double chi2Ref, chi2Opt;
        run(false, pRef, CRef, chi2Ref);
        run(true,  pOpt, COpt, chi2Opt);

        worstP = std::max(worstP, maxRelDiff(pRef, pOpt));
        worstC = std::max(worstC, maxRelDiff(CRef, COpt));
        worstChi2 = std::max(worstChi2,
                             std::fabs(chi2Ref - chi2Opt) / std::max(1e-300, std::fabs(chi2Ref)));
      }
    }
  }

  EXPECT_LT(worstP, kTol)    << "state rel diff = "  << worstP;
  EXPECT_LT(worstC, kTol)    << "cov rel diff = "    << worstC;
  EXPECT_LT(worstChi2, kTol) << "chi2 rel diff = "   << worstChi2;
}

} // namespace genfit
