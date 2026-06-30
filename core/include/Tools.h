/* Copyright 2008-2010, Technische Universitaet Muenchen,
   Authors: Christian Hoeppner & Sebastian Neubert & Johannes Rauch

   This file is part of GENFIT.

   GENFIT is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GENFIT is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with GENFIT.  If not, see <http://www.gnu.org/licenses/>.
*/

/** @addtogroup genfit
 * @{
 */

#ifndef genfit_Tools_h
#define genfit_Tools_h

#include <TVectorD.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>

/**
 * @brief Matrix inversion tools.
 */
namespace genfit {

class AbsHMatrix;

namespace tools {

/** @brief Invert a matrix, throwing an Exception when inversion fails.
 * Optional calculation of determinant.
 */
void invertMatrix(const TMatrixDSym& mat, TMatrixDSym& inv, double* determinant = nullptr);
/** @brief Same, replacing its argument.
 */
void invertMatrix(TMatrixDSym& mat, double* determinant = nullptr);

/** @brief Solves R^t x = b, replacing b with the solution for x.  R is
 *  assumed to be upper diagonal.
 */
bool transposedForwardSubstitution(const TMatrixD& R, TVectorD& b);
/** @brief Same, for a column of the matrix b.  */
bool transposedForwardSubstitution(const TMatrixD& R, TMatrixD& b, int nCol);
/** @brief Inverts the transpose of the upper right matrix R into inv.  */
bool transposedInvert(const TMatrixD& R, TMatrixD& inv);

/** @brief Inverse-variance weighted average of two Gaussian estimates.
 *
 *  Given states @p state1, @p state2 with covariances @p cov1, @p cov2 this
 *  computes the combined estimate
 *      avgCov   = (cov1^-1 + cov2^-1)^-1
 *      avgState = avgCov (cov1^-1 state1 + cov2^-1 state2)
 *  using the numerically-stable square-root (QR) formulation, i.e. without
 *  ever forming cov1^-1 or cov2^-1 explicitly.
 *
 *  @p avgState receives the averaged state.  @p avgCovFactor receives the
 *  lower-triangular factor L of the averaged covariance, so that
 *      avgCov = L^T L = TMatrixDSym(TMatrixDSym::kAtA, avgCovFactor).
 *
 *  Returns false (leaving the outputs unspecified) if either covariance is
 *  not positive definite.
 */
bool averageState(const TVectorD& state1, const TMatrixDSym& cov1,
                  const TVectorD& state2, const TMatrixDSym& cov2,
                  TVectorD& avgState, TMatrixD& avgCovFactor);

/** @brief In-place similarity transform: sym <- B * sym * B^T.
 *
 *  Drop-in replacement for ROOT's TMatrixDSym::Similarity(const TMatrixD&),
 *  specialised for the small dense matrices used by the Kalman fitter.  @p sym
 *  is an (n x n) symmetric matrix and @p B an (m x n) matrix; on return @p sym
 *  holds the (m x m) symmetric product B*sym*B^T (resized if m != n).
 *
 *  The arithmetic is mathematically identical to ROOT's but uses a different
 *  summation order, so the result agrees with TMatrixDSym::Similarity to within
 *  rounding (~1e-12 relative), not necessarily bit-for-bit.
 */
void similarity(const TMatrixD& B, TMatrixDSym& sym);

/** @brief Scalar similarity: returns v^T * A * v.
 *
 *  Drop-in replacement for A.Similarity(const TVectorD&). */
double similarity(const TVectorD& v, const TMatrixDSym& A);

/** @brief Replaces A with an upper right matrix connected to A by
 *  an orthongonal transformation.  I.e., it computes R from a QR
 *  decomposition of A = QR, replacing A.
 */
void QR(TMatrixD& A);

/** @brief Replaces A with an upper right matrix connected to A by
 *  an orthongonal transformation.  I.e., it computes R from a QR
 *  decomposition of A = QR, replacing A.  Also replaces b by Q'b
 *  where Q' is the transposed of Q.
 */
void QR(TMatrixD& A, TVectorD& b);

/** @brief Calculate a sqrt for the positive semidefinite noise
 *  matrix.  Rows corresponding to zero eigenvalues are omitted.
 *  This gives the transposed of the square root, i.e.
 *    noise = noiseSqrt * noiseSqrt'
 */    
void
noiseMatrixSqrt(const TMatrixDSym& noise,
		TMatrixD& noiseSqrt);

/** @brief Calculates the square root of the covariance matrix after
 *  the Kalman prediction (i.e. extrapolation) with transport matrix F
 *  and the noise square root Q.  Gives the new covariance square
 *  root.  */
void
kalmanPredictionCovSqrt(const TMatrixD& S,
			const TMatrixD& F, const TMatrixD& Q,
			TMatrixD& Snew);

/** @brief Calculate the Kalman measurement update with no transport.
 *  x, S : state prediction, covariance square root
 *  res, R, H : residual, measurement covariance square root, H matrix of the measurement
 */
void
kalmanUpdateSqrt(const TMatrixD& S,
		 const TVectorD& res, const TMatrixD& R, const AbsHMatrix* H,
		 TVectorD& update, TMatrixD& SNew);

} /* End of namespace tools */
} /* End of namespace genfit */
/** @} */

#endif // genfit_Tools_h
