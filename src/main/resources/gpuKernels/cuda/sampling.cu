/*
 * BranchMARK* PAC Phase-1 ancestral-sampling CUDA kernel.
 *
 * build_cdf_n_children + draw_cdf_n_children: for one non-leaf lambda edge,
 * build one conditional CDF per distinct parent M-state, then draw one lambda
 * state per sample from the reused CDF. The per-lambda log-weight is exactly the
 * DP upper-bound term  logw(mIdx,lIdx) = -E_min(mIdx,lIdx)*invRT + Σ_child logZUpper,
 * i.e. the upper half of full_dp_n_children. This is statistically equivalent
 * to the CPU inverse-transform sampler, not bit-identical to Java SplittableRandom.
 *
 * sample_gumbel_n_children is kept as a compatibility/debug kernel; the Java
 * runner now uses the CDF kernels so repeated mIdx samples share the expensive
 * log-weight computation.
 *
 * One CUDA block per sample; threads stride over lambda states; block argmax
 * reduction picks the winning lambda. Independent Gumbel noise per (sample,
 * lambda) comes from a stateless counter-based hash (splitmix64) of
 * (baseSeed, sampleSlot, lIdx) — no device RNG state, reproducible per edge.
 *
 * Compile with:
 * nvcc -fatbin
 *   -gencode=arch=compute_70,code=sm_70
 *   -gencode=arch=compute_86,code=sm_86
 *   -gencode=arch=compute_86,code=compute_86
 *   sampling.cu -o sampling.bin
 */

#include <math.h>

#define MAX_EDGE_POSITIONS 32

__device__ void decode_state(long long idx, const int *counts, int count, int *out) {
	for (int i=count - 1; i>=0; i--) {
		int n = counts[i];
		out[i] = (int)(idx % (long long)n);
		idx /= (long long)n;
	}
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
	int lmPairCount
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

/* Sum of one child table over all N children at the current state. */
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

/* Stateless counter-based RNG: splitmix64 finalizer over a mixed counter. */
__device__ unsigned long long splitmix64(unsigned long long x) {
	x += 0x9E3779B97F4A7C15ULL;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
	return x ^ (x >> 31);
}

/* Independent Gumbel(0,1) per (sample, lambda): -log(-log(u)), u in (0,1). */
__device__ double gumbel_noise(unsigned long long baseSeed, int sampleSlot, int lIdx) {
	unsigned long long h = splitmix64(baseSeed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(unsigned int)sampleSlot);
	h = splitmix64(h + (unsigned long long)(unsigned int)lIdx);
	double u = ((double)(h >> 11) + 0.5) * (1.0/9007199254740992.0); /* (0,1), 2^-53 spacing */
	return -log(-log(u));
}

__device__ double uniform01(unsigned long long baseSeed, int sampleSlot) {
	unsigned long long h = splitmix64(baseSeed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(unsigned int)sampleSlot);
	double u = ((double)(h >> 11) + 0.5) * (1.0/9007199254740992.0); /* (0,1), 2^-53 spacing */
	return u;
}

__device__ void argmax_pair(double *score, int *idx, double otherScore, int otherIdx) {
	if (otherScore > *score) {
		*score = otherScore;
		*idx = otherIdx;
	}
}

__device__ void warp_argmax(double *score, int *idx) {
	unsigned int mask = __activemask();
	int lane = threadIdx.x & 31;
	for (int offset=16; offset>0; offset >>= 1) {
		double otherScore = __shfl_down_sync(mask, *score, offset);
		int otherIdx = __shfl_down_sync(mask, *idx, offset);
		if (lane + offset < 32 && (mask & (1u << (lane + offset)))) {
			argmax_pair(score, idx, otherScore, otherIdx);
		}
	}
}

__device__ double warp_max_double(double value) {
	unsigned int mask = __activemask();
	int lane = threadIdx.x & 31;
	for (int offset=16; offset>0; offset >>= 1) {
		double other = __shfl_down_sync(mask, value, offset);
		if (lane + offset < 32 && (mask & (1u << (lane + offset)))) {
			value = fmax(value, other);
		}
	}
	return value;
}

extern "C" __global__ void build_cdf_n_children(
	const int *mCounts,
	const int *lambdaCounts,
	const double *lambdaOnlyMin,
	const double *lmMin,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt,
	const long long *childTableBase,
	const double *childUpperAll,
	const long long *mIdxByGroup,
	double *groupCdf,
	double *groupTotal,
	int numGroups,
	int totalLambdaStates,
	int mCount,
	int lambdaCount,
	int lmPairCount,
	int numChildren,
	double invRT
) {
	int groupSlot = blockIdx.x;
	if (groupSlot >= numGroups) {
		return;
	}

	extern __shared__ double smem[];
	double *warpMaxS = smem;

	int tid = threadIdx.x;
	int lane = tid & 31;
	int warp = tid >> 5;
	int numWarps = (blockDim.x + 31) >> 5;
	int mLocal[MAX_EDGE_POSITIONS];
	int lambdaLocal[MAX_EDGE_POSITIONS];
	long long mIdx = mIdxByGroup[groupSlot];
	long long cdfBase = (long long)groupSlot * (long long)totalLambdaStates;
	decode_state(mIdx, mCounts, mCount, mLocal);

	double localMax = -INFINITY;
	for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
		double eMin = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyMin, lmMin, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount);
		double fUpper = child_sum(childUpperAll, childTableBase, numChildren, mLocal, lambdaLocal,
			childMSrcAll, childMStrideAll, childMTermOff, childMTermCnt,
			childLSrcAll, childLStrideAll, childLTermOff, childLTermCnt);
		double logWeight = -eMin*invRT + fUpper;
		groupCdf[cdfBase + lIdx] = logWeight;
		if (!isnan(logWeight)) {
			localMax = fmax(localMax, logWeight);
		}
	}

	localMax = warp_max_double(localMax);
	if (lane == 0) {
		warpMaxS[warp] = localMax;
	}
	__syncthreads();

	if (warp == 0) {
		double blockMax = (lane < numWarps) ? warpMaxS[lane] : -INFINITY;
		blockMax = warp_max_double(blockMax);
		if (lane == 0) {
			warpMaxS[0] = blockMax;
		}
	}
	__syncthreads();

	double maxLog = warpMaxS[0];
	if (isfinite(maxLog)) {
		for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
			double weight = exp(groupCdf[cdfBase + lIdx] - maxLog);
			groupCdf[cdfBase + lIdx] = isfinite(weight) ? weight : 0.0;
		}
	} else {
		for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
			groupCdf[cdfBase + lIdx] = 0.0;
		}
	}
	__syncthreads();

	if (tid == 0) {
		double running = 0.0;
		for (int lIdx=0; lIdx<totalLambdaStates; lIdx++) {
			running += groupCdf[cdfBase + lIdx];
			groupCdf[cdfBase + lIdx] = running;
		}
		groupTotal[groupSlot] = running;
	}
}

extern "C" __global__ void draw_cdf_n_children(
	const double *groupCdf,
	const double *groupTotal,
	const int *groupForSample,
	const int *sampleSlots,
	int *outLIdx,
	int numSamples,
	int totalLambdaStates,
	unsigned long long baseSeed
) {
	int localSample = blockIdx.x*blockDim.x + threadIdx.x;
	if (localSample >= numSamples) {
		return;
	}

	int groupSlot = groupForSample[localSample];
	double total = groupTotal[groupSlot];
	if (!(total > 0.0) || !isfinite(total)) {
		outLIdx[localSample] = totalLambdaStates - 1;
		return;
	}

	int sampleSlot = sampleSlots[localSample];
	double target = uniform01(baseSeed, sampleSlot) * total;
	long long cdfBase = (long long)groupSlot * (long long)totalLambdaStates;
	int lo = 0;
	int hi = totalLambdaStates - 1;
	while (lo < hi) {
		int mid = (lo + hi) >> 1;
		if (target < groupCdf[cdfBase + mid]) {
			hi = mid;
		} else {
			lo = mid + 1;
		}
	}
	outLIdx[localSample] = lo;
}

extern "C" __global__ void sample_gumbel_n_children(
	const int *mCounts,
	const int *lambdaCounts,
	const double *lambdaOnlyMin,
	const double *lmMin,
	const int *lmLamSlots,
	const int *lmMSlots,
	const int *lmMCounts,
	const long long *lmOffsets,
	const int *childMSrcAll,
	const long long *childMStrideAll,
	const int *childMTermOff,
	const int *childMTermCnt,
	const int *childLSrcAll,
	const long long *childLStrideAll,
	const int *childLTermOff,
	const int *childLTermCnt,
	const long long *childTableBase,
	const double *childUpperAll,
	const long long *mIdxPerSample,
	int *outLIdx,
	int numSamples,
	int totalLambdaStates,
	int mCount,
	int lambdaCount,
	int lmPairCount,
	int numChildren,
	double invRT,
	unsigned long long baseSeed
) {
	int sampleSlot = blockIdx.x;
	if (sampleSlot >= numSamples) {
		return;
	}
	long long mIdx = mIdxPerSample[sampleSlot];

	extern __shared__ double smem[];
	double *warpScoreS = smem;                       /* ceil(blockDim.x/32) doubles */
	int *warpIdxS = (int*)(warpScoreS + ((blockDim.x + 31) >> 5));

	int tid = threadIdx.x;
	int lane = tid & 31;
	int warp = tid >> 5;
	int numWarps = (blockDim.x + 31) >> 5;
	int mLocal[MAX_EDGE_POSITIONS];
	int lambdaLocal[MAX_EDGE_POSITIONS];
	decode_state(mIdx, mCounts, mCount, mLocal);

	double bestScore = -INFINITY;
	int bestIdx = totalLambdaStates - 1;   /* matches CPU streaming default */
	for (int lIdx=tid; lIdx<totalLambdaStates; lIdx += blockDim.x) {
		double eMin = local_energy(lIdx, mLocal, lambdaLocal, lambdaCounts, lambdaCount,
			lambdaOnlyMin, lmMin, lmLamSlots, lmMSlots, lmMCounts, lmOffsets, lmPairCount);
		double fUpper = child_sum(childUpperAll, childTableBase, numChildren, mLocal, lambdaLocal,
			childMSrcAll, childMStrideAll, childMTermOff, childMTermCnt,
			childLSrcAll, childLStrideAll, childLTermOff, childLTermCnt);
		double logWeight = -eMin*invRT + fUpper;
		if (isnan(logWeight)) {
			continue;   /* CPU skips NaN log-weights */
		}
		double score = logWeight + gumbel_noise(baseSeed, sampleSlot, lIdx);
		if (score > bestScore) {
			bestScore = score;
			bestIdx = lIdx;
		}
	}

	warp_argmax(&bestScore, &bestIdx);
	if (lane == 0) {
		warpScoreS[warp] = bestScore;
		warpIdxS[warp] = bestIdx;
	}
	__syncthreads();

	if (warp == 0) {
		double blockBestScore = (lane < numWarps) ? warpScoreS[lane] : -INFINITY;
		int blockBestIdx = (lane < numWarps) ? warpIdxS[lane] : totalLambdaStates - 1;
		warp_argmax(&blockBestScore, &blockBestIdx);
		if (lane == 0) {
			outLIdx[sampleSlot] = blockBestIdx;
		}
	}
}
