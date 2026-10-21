/*
 * BranchMARK* full-DP CUDA kernels.
 *
 * full_dp_n_children: one non-leaf edge with N (>=1) F-set children, dense
 * parent/child DP tables, full lambda enumeration. Per-child fold plans are
 * concatenated CSR-style (per-child term offsets/counts + per-child table base).
 * One CUDA block per parent M-state; mStateOffset + local output indexing lets
 * the M-state range be split across multiple GPUs.
 *
 * Compile with:
 * nvcc -fatbin
 *   -gencode=arch=compute_86,code=sm_86
 *   -gencode=arch=compute_86,code=compute_86
 *   dp.cu -o dp.bin
 */

#include <math.h>

#define MAX_EDGE_POSITIONS 32
#define MAX_CHILDREN 64

/* Online log-sum-exp: add one term x to a running (max=*m, sum=*s) where the
 * invariant is s == sum_i exp(term_i - m). Single-pass replacement for the old
 * max-pass + exp-pass. NaN in any term poisons the row (matches old max_nan);
 * -inf terms contribute nothing. Invariant: m == -inf  <=>  s == 0 (no finite
 * term yet), so the rescale never hits exp(-inf - -inf). */
__device__ void lse_update(double *m, double *s, double x) {
	if (isnan(x)) { *m = NAN; *s = NAN; return; }
	if (x == -INFINITY) return;
	if (isnan(*m)) return;
	if (x > *m) {
		*s = (*m == -INFINITY) ? 1.0 : (*s) * exp(*m - x) + 1.0;
		*m = x;
	} else {
		*s += exp(x - *m);
	}
}

/* Merge another stable log-sum-exp partial into (*m,*s). Both sums are
 * expressed relative to their corresponding maxima. This is used by the
 * fully out-of-core kernel to combine disjoint lambda tiles without ever
 * materialising per-lambda output values. */
__device__ void lse_merge_partial(double *m, double *s, double otherM, double otherS) {
	if (isnan(*m) || isnan(*s) || isnan(otherM) || isnan(otherS)) {
		*m = NAN;
		*s = NAN;
		return;
	}
	if (otherM == -INFINITY) return;
	if (*m == -INFINITY) {
		*m = otherM;
		*s = otherS;
		return;
	}
	double mergedM = fmax(*m, otherM);
	*s = (*s) * exp(*m - mergedM) + otherS * exp(otherM - mergedM);
	*m = mergedM;
}

/* NaN-poisoning max for the block max-reduce: a NaN energy anywhere makes the
 * whole row NaN (matches the Java reference). */
__device__ double max_nan(double a, double b) {
	if (isnan(a) || isnan(b)) { return NAN; }
	return fmax(a, b);
}

__device__ void decode_state(long long idx, const int *counts, int count, int *out) {
	for (int i=count - 1; i>=0; i--) {
		int n = counts[i];
		out[i] = (int)(idx % (long long)n);
		idx /= (long long)n;
	}
}

/* Exact edge-local three-position factors.  A slot below mCount addresses
 * mLocal; all other slots address lambdaLocal after subtracting mCount. */
__device__ double triple_energy(
	const int *mLocal,
	const int *lambdaLocal,
	int mCount,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleValues,
	int tripleFactorCount
) {
	double e = 0.0;
	for (int factor=0; factor<tripleFactorCount; factor++) {
		long long index = tripleOffsets[factor];
		int meta = factor*3;
		for (int term=0; term<3; term++) {
			int slot = tripleSlots[meta + term];
			int rc = slot < mCount
				? mLocal[slot]
				: lambdaLocal[slot - mCount];
			index += (long long)rc*tripleStrides[meta + term];
		}
		e += tripleValues[index];
	}
	return e;
}

__device__ double local_energy(
	int lIdx,
	const int *mLocal,
	int *lambdaLocal,
	const int *lambdaCounts,
	int lambdaCount,
	const double *lambdaOnly,
	const double *lmPairs,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	int lmPairCount,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleValues,
	int tripleFactorCount,
	int mCount
) {
	decode_state((long long)lIdx, lambdaCounts, lambdaCount, lambdaLocal);

	double e = lambdaOnly[lIdx];
	for (int p=0; p<lmPairCount; p++) {
		int lamSlot = lmLamSlots[p];
		int mSlot = lmMSlots[p];
		long long off = lmOffsets[p]
			+ (long long)lambdaLocal[lamSlot]*(long long)lmMCounts[p]
			+ (long long)mLocal[mSlot];
		e += lmPairs[off];
	}
	e += triple_energy(mLocal, lambdaLocal, mCount,
		tripleSlots, tripleStrides, tripleOffsets, tripleValues,
		tripleFactorCount);
	return e;
}

/* local_energy() variant for a lambda tile. The full mixed-radix lambda index
 * is still decoded so LM and child projections keep their original meaning,
 * while lambdaOnlyValue comes from the bounded tile buffer. */
__device__ double local_energy_indexed(
	int globalLIdx,
	double lambdaOnlyValue,
	const int *mLocal,
	int *lambdaLocal,
	const int *lambdaCounts,
	int lambdaCount,
	const double *lmPairs,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	int lmPairCount,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleValues,
	int tripleFactorCount,
	int mCount
) {
	decode_state((long long)globalLIdx, lambdaCounts, lambdaCount, lambdaLocal);
	double e = lambdaOnlyValue;
	for (int p=0; p<lmPairCount; p++) {
		int lamSlot = lmLamSlots[p];
		int mSlot = lmMSlots[p];
		long long off = lmOffsets[p]
			+ (long long)lambdaLocal[lamSlot]*(long long)lmMCounts[p]
			+ (long long)mLocal[mSlot];
		e += lmPairs[off];
	}
	e += triple_energy(mLocal, lambdaLocal, mCount,
		tripleSlots, tripleStrides, tripleOffsets, tripleValues,
		tripleFactorCount);
	return e;
}

/* Mixed-radix child index for one child slice (M terms then lambda terms). */
__device__ long long child_index_slice(
	const int *mLocal,
	const int *lambdaLocal,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	int mOff, int mCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	int lOff, int lCnt
) {
	long long idx = 0;
	for (int t=0; t<mCnt; t++) {
		idx += (long long)mLocal[childMSrcAll[mOff + t]]*childMStrideAll[mOff + t];
	}
	for (int t=0; t<lCnt; t++) {
		idx += (long long)lambdaLocal[childLSrcAll[lOff + t]]*childLStrideAll[lOff + t];
	}
	return idx;
}

__device__ long long find_child_row(
	long long key,
	const long long *childRowKeysAll,
	long long base,
	int count
) {
	int lo = 0;
	int hi = count - 1;
	while (lo <= hi) {
		int mid = lo + ((hi - lo) >> 1);
		long long value = childRowKeysAll[base + (long long)mid];
		if (value < key) {
			lo = mid + 1;
		} else if (value > key) {
			hi = mid - 1;
		} else {
			return (long long)mid;
		}
	}
	return -1;
}

/* Sum of one child table (childTable = lower or upper) over all N children
 * at the current (mLocal, lambdaLocal) state. */
__device__ double child_sum(
	const double *childTable,
	const long long *childTableBase,
	int numChildren,
	const int *mLocal,
	const int *lambdaLocal,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt
) {
	double s = 0.0;
	for (int c=0; c<numChildren; c++) {
		long long fIdx = child_index_slice(mLocal, lambdaLocal,
			childMSrcAll, childMStrideAll, childMTermOff[c], childMTermCnt[c],
			childLSrcAll, childLStrideAll, childLTermOff[c], childLTermCnt[c]);
		s += childTable[childTableBase[c] + fIdx];
	}
	return s;
}

__device__ double child_sum_sliced(
	const double *childTable,
	const long long *childPackedBase,
	const long long *childPackedRows,
	const long long *childLambdaStates,
	int numChildren,
	const int *lambdaLocal,
	const int *childLSrcAll,
	const long long *childLPackedStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt
) {
	double s = 0.0;
	for (int c=0; c<numChildren; c++) {
		long long lIdx = 0;
		int lOff = childLTermOff[c];
		int lCnt = childLTermCnt[c];
		for (int t=0; t<lCnt; t++) {
			lIdx += (long long)lambdaLocal[childLSrcAll[lOff + t]]*childLPackedStrideAll[lOff + t];
		}
		s += childTable[childPackedBase[c] + childPackedRows[c]*childLambdaStates[c] + lIdx];
	}
	return s;
}

/* Child lookup for a rectangular multi-child row x lambda tile. Row and
 * lambda projections are independently packed per child, so neither a whole
 * child nor one complete child row must fit in VRAM. */
__device__ double child_sum_out_of_core(
	const double *childTable,
	const long long *childPackedBase,
	const long long *childPackedRows,
	const long long *childLambdaKeyBase,
	const int *childLambdaKeyCount,
	const long long *childLambdaKeysAll,
	int numChildren,
	const int *lambdaLocal,
	const int *childLSrcAll,
	const long long *childLPackedStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt
) {
	double s = 0.0;
	for (int c=0; c<numChildren; c++) {
		long long lambdaKey = 0;
		int lOff = childLTermOff[c];
		int lCnt = childLTermCnt[c];
		for (int t=0; t<lCnt; t++) {
			lambdaKey += (long long)lambdaLocal[childLSrcAll[lOff + t]]
				* childLPackedStrideAll[lOff + t];
		}
		long long packedLambda = find_child_row(lambdaKey,
			childLambdaKeysAll, childLambdaKeyBase[c], childLambdaKeyCount[c]);
		if (packedLambda < 0) return NAN;
		s += childTable[childPackedBase[c]
			+ childPackedRows[c]*(long long)childLambdaKeyCount[c]
			+ packedLambda];
	}
	return s;
}

__device__ double child_sum_hybrid(
	const double *residentTable,
	const double *streamedTable,
	const long long *residentTableBase,
	const long long *childLambdaStates,
	int streamedChild,
	long long streamedPackedRow,
	int numChildren,
	const int *mLocal,
	const int *lambdaLocal,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	const long long *childLPackedStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt
) {
	double s = 0.0;
	for (int c=0; c<numChildren; c++) {
		if (c == streamedChild) {
			long long lIdx = 0;
			int lOff = childLTermOff[c];
			int lCnt = childLTermCnt[c];
			for (int t=0; t<lCnt; t++) {
				lIdx += (long long)lambdaLocal[childLSrcAll[lOff + t]]
					* childLPackedStrideAll[lOff + t];
			}
			s += streamedTable[streamedPackedRow*childLambdaStates[c] + lIdx];
		} else {
			long long fIdx = child_index_slice(mLocal, lambdaLocal,
				childMSrcAll, childMStrideAll, childMTermOff[c], childMTermCnt[c],
				childLSrcAll, childLStrideAll, childLTermOff[c], childLTermCnt[c]);
			s += residentTable[residentTableBase[c] + fIdx];
		}
	}
	return s;
}

extern "C" __global__ void full_dp_n_children(
	const int *mCounts,
	const int *lambdaCounts,
	const double *lambdaOnlyRigid,
	const double *lambdaOnlyMin,
	const double *lmRigid,
	const double *lmMin,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleRigid,
	const double *tripleMin,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt,
	const long long *childTableBase,
	const double *childLowerAll,
	const double *childUpperAll,
	double *outLower,
	double *outUpper,
	long long mStateCount,
	long long mStateOffset,
	int totalLambdaStates,
	int mCount,
	int lambdaCount,
	int lmPairCount,
	int tripleFactorCount,
	int numChildren,
	double invRT
) {
	long long localIdx = (long long)blockIdx.x;
	long long mIdx = mStateOffset + localIdx;
	if (mIdx >= mStateCount) {
		return;
	}

	extern __shared__ double scratch[];
	double *lowerMax = scratch;
	double *upperMax = lowerMax + blockDim.x;
	double *lowerSum = upperMax + blockDim.x;
	double *upperSum = lowerSum + blockDim.x;

	int tid = threadIdx.x;
	int mLocal[MAX_EDGE_POSITIONS];
	int lambdaLocal[MAX_EDGE_POSITIONS];

	decode_state(mIdx, mCounts, mCount, mLocal);

	/* Single pass: one (max,sum) running pair per thread, child tables + decode
	 * touched once (was twice). The old max-pass then exp-pass are fused. */
	double tLowerMax = -INFINITY, tLowerSum = 0.0;
	double tUpperMax = -INFINITY, tUpperSum = 0.0;
	for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
		double eRigid = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyRigid, lmRigid, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount,
			tripleSlots, tripleStrides, tripleOffsets, tripleRigid, tripleFactorCount, mCount);
		double fLower = child_sum(childLowerAll, childTableBase, numChildren, mLocal, lambdaLocal,
			childMSrcAll, childMStrideAll, childMTermOff, childMTermCnt,
			childLSrcAll, childLStrideAll, childLTermOff, childLTermCnt);
		double eMin = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyMin, lmMin, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount,
			tripleSlots, tripleStrides, tripleOffsets, tripleMin, tripleFactorCount, mCount);
		double fUpper = child_sum(childUpperAll, childTableBase, numChildren, mLocal, lambdaLocal,
			childMSrcAll, childMStrideAll, childMTermOff, childMTermCnt,
			childLSrcAll, childLStrideAll, childLTermOff, childLTermCnt);

		lse_update(&tLowerMax, &tLowerSum, -eRigid*invRT + fLower);
		lse_update(&tUpperMax, &tUpperSum, -eMin*invRT + fUpper);
	}

	/* Block reduction in two cheap halves to minimise fp64 exp (A5000 has no
	 * fp64 SFU, so exp is the costly op): max-reduce (no exp), then exactly ONE
	 * rescale per thread, then a plain additive sum-reduce. A full (max,sum)
	 * merge tree would instead pay 2 exp per combine (~8x more exp here). */
	lowerMax[tid] = tLowerMax;
	upperMax[tid] = tUpperMax;
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerMax[tid] = max_nan(lowerMax[tid], lowerMax[tid + stride]);
			upperMax[tid] = max_nan(upperMax[tid], upperMax[tid + stride]);
		}
		__syncthreads();
	}
	double gLowerMax = lowerMax[0];
	double gUpperMax = upperMax[0];

	/* Rescale each thread's running sum from its local max to the block max.
	 * gMax == -inf means no finite term in the whole row -> sum stays 0. A finite
	 * thread paired with gMax == NaN yields NaN, which the final guard maps to NaN. */
	lowerSum[tid] = (gLowerMax == -INFINITY) ? 0.0 : tLowerSum * exp(tLowerMax - gLowerMax);
	upperSum[tid] = (gUpperMax == -INFINITY) ? 0.0 : tUpperSum * exp(tUpperMax - gUpperMax);
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerSum[tid] += lowerSum[tid + stride];
			upperSum[tid] += upperSum[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		outLower[localIdx] = (gLowerMax == -INFINITY || isnan(gLowerMax))
			? gLowerMax
			: gLowerMax + log(lowerSum[0]);
		outUpper[localIdx] = (gUpperMax == -INFINITY || isnan(gUpperMax))
			? gUpperMax
			: gUpperMax + log(upperSum[0]);
	}
}

extern "C" __global__ void full_dp_n_children_sliced(
	const int *mCounts,
	const int *lambdaCounts,
	const double *lambdaOnlyRigid,
	const double *lambdaOnlyMin,
	const double *lmRigid,
	const double *lmMin,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleRigid,
	const double *tripleMin,
	const int *childMSrcAll,
	const long long *childMPackedStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLPackedStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt,
	const long long *childPackedBase,
	const long long *childRowKeyBase,
	const int *childRowKeyCount,
	const long long *childRowKeysAll,
	const long long *childLambdaStates,
	const double *childLowerPacked,
	const double *childUpperPacked,
	const long long *mIdxList,
	double *outLower,
	double *outUpper,
	long long mStateCount,
	int totalLambdaStates,
	int mCount,
	int lambdaCount,
	int lmPairCount,
	int tripleFactorCount,
	int numChildren,
	double invRT
) {
	long long localIdx = (long long)blockIdx.x;
	long long mIdx = mIdxList[localIdx];
	if (mIdx >= mStateCount) {
		return;
	}

	extern __shared__ double scratch[];
	double *lowerMax = scratch;
	double *upperMax = lowerMax + blockDim.x;
	double *lowerSum = upperMax + blockDim.x;
	double *upperSum = lowerSum + blockDim.x;

	int tid = threadIdx.x;
	int mLocal[MAX_EDGE_POSITIONS];
	int lambdaLocal[MAX_EDGE_POSITIONS];
	long long childPackedRows[MAX_CHILDREN];

	decode_state(mIdx, mCounts, mCount, mLocal);

	bool rowOk = true;
	if (numChildren > MAX_CHILDREN) {
		rowOk = false;
	}
	for (int c=0; c<numChildren && c<MAX_CHILDREN; c++) {
		long long rowKey = 0;
		int mOff = childMTermOff[c];
		int mCnt = childMTermCnt[c];
		for (int t=0; t<mCnt; t++) {
			rowKey += (long long)mLocal[childMSrcAll[mOff + t]]*childMPackedStrideAll[mOff + t];
		}
		long long packedRow = find_child_row(rowKey, childRowKeysAll, childRowKeyBase[c], childRowKeyCount[c]);
		childPackedRows[c] = packedRow;
		if (packedRow < 0) {
			rowOk = false;
		}
	}
	if (!rowOk) {
		if (tid == 0) {
			outLower[localIdx] = NAN;
			outUpper[localIdx] = NAN;
		}
		return;
	}

	double tLowerMax = -INFINITY, tLowerSum = 0.0;
	double tUpperMax = -INFINITY, tUpperSum = 0.0;
	for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
		double eRigid = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyRigid, lmRigid, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount,
			tripleSlots, tripleStrides, tripleOffsets, tripleRigid, tripleFactorCount, mCount);
		double fLower = child_sum_sliced(childLowerPacked, childPackedBase, childPackedRows, childLambdaStates,
			numChildren, lambdaLocal, childLSrcAll, childLPackedStrideAll, childLTermOff, childLTermCnt);
		double eMin = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyMin, lmMin, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount,
			tripleSlots, tripleStrides, tripleOffsets, tripleMin, tripleFactorCount, mCount);
		double fUpper = child_sum_sliced(childUpperPacked, childPackedBase, childPackedRows, childLambdaStates,
			numChildren, lambdaLocal, childLSrcAll, childLPackedStrideAll, childLTermOff, childLTermCnt);

		lse_update(&tLowerMax, &tLowerSum, -eRigid*invRT + fLower);
		lse_update(&tUpperMax, &tUpperSum, -eMin*invRT + fUpper);
	}

	lowerMax[tid] = tLowerMax;
	upperMax[tid] = tUpperMax;
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerMax[tid] = max_nan(lowerMax[tid], lowerMax[tid + stride]);
			upperMax[tid] = max_nan(upperMax[tid], upperMax[tid + stride]);
		}
		__syncthreads();
	}
	double gLowerMax = lowerMax[0];
	double gUpperMax = upperMax[0];

	lowerSum[tid] = (gLowerMax == -INFINITY) ? 0.0 : tLowerSum * exp(tLowerMax - gLowerMax);
	upperSum[tid] = (gUpperMax == -INFINITY) ? 0.0 : tUpperSum * exp(tUpperMax - gUpperMax);
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerSum[tid] += lowerSum[tid + stride];
			upperSum[tid] += upperSum[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		outLower[localIdx] = (gLowerMax == -INFINITY || isnan(gLowerMax))
			? gLowerMax
			: gLowerMax + log(lowerSum[0]);
		outUpper[localIdx] = (gUpperMax == -INFINITY || isnan(gUpperMax))
			? gUpperMax
			: gUpperMax + log(upperSum[0]);
	}
}

/* Fully bounded fallback. One launch consumes one rectangular lambda tile for
 * one output-M block. Repeated launches merge stable (max,sumExp) partials in
 * acc*, and only the final lambda tile materialises logZ in out*. */
extern "C" __global__ void full_dp_n_children_out_of_core(
	const int *mCounts,
	const int *lambdaCounts,
	const double *lambdaOnlyRigidTile,
	const double *lambdaOnlyMinTile,
	const double *lmRigid,
	const double *lmMin,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleRigid,
	const double *tripleMin,
	const int *childMSrcAll,
	const long long *childMPackedStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLPackedStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt,
	const long long *childPackedBase,
	const long long *childRowKeyBase,
	const int *childRowKeyCount,
	const long long *childRowKeysAll,
	const long long *childLambdaKeyBase,
	const int *childLambdaKeyCount,
	const long long *childLambdaKeysAll,
	const double *childLowerPacked,
	const double *childUpperPacked,
	const int *lambdaIdxList,
	const long long *mIdxList,
	double *accLowerMax,
	double *accLowerSum,
	double *accUpperMax,
	double *accUpperSum,
	double *outLower,
	double *outUpper,
	long long mStateCount,
	int lambdaTileStates,
	int mCount,
	int lambdaCount,
	int lmPairCount,
	int tripleFactorCount,
	int numChildren,
	int firstLambdaTile,
	int lastLambdaTile,
	double invRT
) {
	long long localIdx = (long long)blockIdx.x;
	long long mIdx = mIdxList[localIdx];
	if (mIdx >= mStateCount) return;

	extern __shared__ double scratch[];
	double *lowerMax = scratch;
	double *upperMax = lowerMax + blockDim.x;
	double *lowerSum = upperMax + blockDim.x;
	double *upperSum = lowerSum + blockDim.x;

	int tid = threadIdx.x;
	int mLocal[MAX_EDGE_POSITIONS];
	int lambdaLocal[MAX_EDGE_POSITIONS];
	long long childPackedRows[MAX_CHILDREN];
	decode_state(mIdx, mCounts, mCount, mLocal);

	bool rowOk = numChildren <= MAX_CHILDREN;
	for (int c=0; c<numChildren && c<MAX_CHILDREN; c++) {
		long long rowKey = 0;
		int mOff = childMTermOff[c];
		int mCnt = childMTermCnt[c];
		for (int t=0; t<mCnt; t++) {
			rowKey += (long long)mLocal[childMSrcAll[mOff + t]]
				* childMPackedStrideAll[mOff + t];
		}
		long long packedRow = find_child_row(rowKey, childRowKeysAll,
			childRowKeyBase[c], childRowKeyCount[c]);
		childPackedRows[c] = packedRow;
		if (packedRow < 0) rowOk = false;
	}
	if (!rowOk) {
		if (tid == 0) {
			accLowerMax[localIdx] = NAN;
			accLowerSum[localIdx] = NAN;
			accUpperMax[localIdx] = NAN;
			accUpperSum[localIdx] = NAN;
			if (lastLambdaTile) {
				outLower[localIdx] = NAN;
				outUpper[localIdx] = NAN;
			}
		}
		return;
	}

	double tLowerMax = -INFINITY, tLowerSum = 0.0;
	double tUpperMax = -INFINITY, tUpperSum = 0.0;
	for (int tileLIdx=tid; tileLIdx<lambdaTileStates;
		tileLIdx += blockDim.x) {
		int globalLIdx = lambdaIdxList[tileLIdx];
		double eRigid = local_energy_indexed(globalLIdx,
			lambdaOnlyRigidTile[tileLIdx], mLocal, lambdaLocal,
			lambdaCounts, lambdaCount, lmRigid, lmLamSlots, lmMSlots,
			lmMCounts, lmOffsets, lmPairCount, tripleSlots, tripleStrides,
			tripleOffsets, tripleRigid, tripleFactorCount, mCount);
		double fLower = child_sum_out_of_core(childLowerPacked,
			childPackedBase, childPackedRows, childLambdaKeyBase,
			childLambdaKeyCount, childLambdaKeysAll, numChildren,
			lambdaLocal, childLSrcAll, childLPackedStrideAll,
			childLTermOff, childLTermCnt);
		double eMin = local_energy_indexed(globalLIdx,
			lambdaOnlyMinTile[tileLIdx], mLocal, lambdaLocal,
			lambdaCounts, lambdaCount, lmMin, lmLamSlots, lmMSlots,
			lmMCounts, lmOffsets, lmPairCount, tripleSlots, tripleStrides,
			tripleOffsets, tripleMin, tripleFactorCount, mCount);
		double fUpper = child_sum_out_of_core(childUpperPacked,
			childPackedBase, childPackedRows, childLambdaKeyBase,
			childLambdaKeyCount, childLambdaKeysAll, numChildren,
			lambdaLocal, childLSrcAll, childLPackedStrideAll,
			childLTermOff, childLTermCnt);

		lse_update(&tLowerMax, &tLowerSum, -eRigid*invRT + fLower);
		lse_update(&tUpperMax, &tUpperSum, -eMin*invRT + fUpper);
	}

	lowerMax[tid] = tLowerMax;
	upperMax[tid] = tUpperMax;
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerMax[tid] = max_nan(lowerMax[tid], lowerMax[tid + stride]);
			upperMax[tid] = max_nan(upperMax[tid], upperMax[tid + stride]);
		}
		__syncthreads();
	}
	double gLowerMax = lowerMax[0];
	double gUpperMax = upperMax[0];
	lowerSum[tid] = (gLowerMax == -INFINITY) ? 0.0
		: tLowerSum * exp(tLowerMax - gLowerMax);
	upperSum[tid] = (gUpperMax == -INFINITY) ? 0.0
		: tUpperSum * exp(tUpperMax - gUpperMax);
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerSum[tid] += lowerSum[tid + stride];
			upperSum[tid] += upperSum[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		double mergedLowerMax = gLowerMax;
		double mergedLowerSum = lowerSum[0];
		double mergedUpperMax = gUpperMax;
		double mergedUpperSum = upperSum[0];
		if (!firstLambdaTile) {
			double oldLowerMax = accLowerMax[localIdx];
			double oldLowerSum = accLowerSum[localIdx];
			double oldUpperMax = accUpperMax[localIdx];
			double oldUpperSum = accUpperSum[localIdx];
			lse_merge_partial(&oldLowerMax, &oldLowerSum,
				mergedLowerMax, mergedLowerSum);
			lse_merge_partial(&oldUpperMax, &oldUpperSum,
				mergedUpperMax, mergedUpperSum);
			mergedLowerMax = oldLowerMax;
			mergedLowerSum = oldLowerSum;
			mergedUpperMax = oldUpperMax;
			mergedUpperSum = oldUpperSum;
		}
		accLowerMax[localIdx] = mergedLowerMax;
		accLowerSum[localIdx] = mergedLowerSum;
		accUpperMax[localIdx] = mergedUpperMax;
		accUpperSum[localIdx] = mergedUpperSum;
		if (lastLambdaTile) {
			outLower[localIdx] = (mergedLowerMax == -INFINITY
				|| isnan(mergedLowerMax)) ? mergedLowerMax
				: mergedLowerMax + log(mergedLowerSum);
			outUpper[localIdx] = (mergedUpperMax == -INFINITY
				|| isnan(mergedUpperMax)) ? mergedUpperMax
				: mergedUpperMax + log(mergedUpperSum);
		}
	}
}

extern "C" __global__ void full_dp_n_children_hybrid(
	const int *mCounts,
	const int *lambdaCounts,
	const double *lambdaOnlyRigid,
	const double *lambdaOnlyMin,
	const double *lmRigid,
	const double *lmMin,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	const int *tripleSlots,
	const long long *tripleStrides,
	const long long *tripleOffsets,
	const double *tripleRigid,
	const double *tripleMin,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	const long long *childMPackedStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	const long long *childLPackedStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt,
	const long long *residentTableBase,
	const long long *childLambdaStates,
	const double *residentLower,
	const double *residentUpper,
	const double *streamedLower,
	const double *streamedUpper,
	const long long *mIdxList,
	double *outLower,
	double *outUpper,
	long long mStateCount,
	int totalLambdaStates,
	int mCount,
	int lambdaCount,
	int lmPairCount,
	int tripleFactorCount,
	int numChildren,
	int streamedChild,
	long long streamedRowStart,
	int streamedRowCount,
	double invRT
) {
	long long localIdx = (long long)blockIdx.x;
	long long mIdx = mIdxList[localIdx];
	if (mIdx >= mStateCount) {
		return;
	}

	extern __shared__ double scratch[];
	double *lowerMax = scratch;
	double *upperMax = lowerMax + blockDim.x;
	double *lowerSum = upperMax + blockDim.x;
	double *upperSum = lowerSum + blockDim.x;

	int tid = threadIdx.x;
	int mLocal[MAX_EDGE_POSITIONS];
	int lambdaLocal[MAX_EDGE_POSITIONS];
	decode_state(mIdx, mCounts, mCount, mLocal);

	long long rowKey = 0;
	int streamMOff = childMTermOff[streamedChild];
	int streamMCnt = childMTermCnt[streamedChild];
	for (int t=0; t<streamMCnt; t++) {
		rowKey += (long long)mLocal[childMSrcAll[streamMOff + t]]
			* childMPackedStrideAll[streamMOff + t];
	}
	long long packedRow = rowKey - streamedRowStart;
	if (packedRow < 0 || packedRow >= (long long)streamedRowCount) {
		if (tid == 0) {
			outLower[localIdx] = NAN;
			outUpper[localIdx] = NAN;
		}
		return;
	}

	double tLowerMax = -INFINITY, tLowerSum = 0.0;
	double tUpperMax = -INFINITY, tUpperSum = 0.0;
	for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
		double eRigid = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyRigid, lmRigid, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount,
			tripleSlots, tripleStrides, tripleOffsets, tripleRigid, tripleFactorCount, mCount);
		double fLower = child_sum_hybrid(residentLower, streamedLower,
			residentTableBase, childLambdaStates, streamedChild, packedRow, numChildren,
			mLocal, lambdaLocal, childMSrcAll, childMStrideAll,
			childMTermOff, childMTermCnt, childLSrcAll, childLStrideAll,
			childLPackedStrideAll, childLTermOff, childLTermCnt);
		double eMin = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyMin, lmMin, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount,
			tripleSlots, tripleStrides, tripleOffsets, tripleMin, tripleFactorCount, mCount);
		double fUpper = child_sum_hybrid(residentUpper, streamedUpper,
			residentTableBase, childLambdaStates, streamedChild, packedRow, numChildren,
			mLocal, lambdaLocal, childMSrcAll, childMStrideAll,
			childMTermOff, childMTermCnt, childLSrcAll, childLStrideAll,
			childLPackedStrideAll, childLTermOff, childLTermCnt);

		lse_update(&tLowerMax, &tLowerSum, -eRigid*invRT + fLower);
		lse_update(&tUpperMax, &tUpperSum, -eMin*invRT + fUpper);
	}

	lowerMax[tid] = tLowerMax;
	upperMax[tid] = tUpperMax;
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerMax[tid] = max_nan(lowerMax[tid], lowerMax[tid + stride]);
			upperMax[tid] = max_nan(upperMax[tid], upperMax[tid + stride]);
		}
		__syncthreads();
	}
	double gLowerMax = lowerMax[0];
	double gUpperMax = upperMax[0];

	lowerSum[tid] = (gLowerMax == -INFINITY) ? 0.0 : tLowerSum * exp(tLowerMax - gLowerMax);
	upperSum[tid] = (gUpperMax == -INFINITY) ? 0.0 : tUpperSum * exp(tUpperMax - gUpperMax);
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
		if (tid < stride) {
			lowerSum[tid] += lowerSum[tid + stride];
			upperSum[tid] += upperSum[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		outLower[localIdx] = (gLowerMax == -INFINITY || isnan(gLowerMax))
			? gLowerMax
			: gLowerMax + log(lowerSum[0]);
		outUpper[localIdx] = (gUpperMax == -INFINITY || isnan(gUpperMax))
			? gUpperMax
			: gUpperMax + log(upperSum[0]);
	}
}
