package edu.duke.cs.osprey.packstar;

import java.util.*;

/**
 * Graph-preserving proposal fitting from adaptation data.
 *
 * <p>Residuals are target CCD energies minus proposal energies. Constant,
 * unary, and retained-pair corrections fit E_eta without changing the target
 * E_t. Source importance weights account for the proposal that generated each
 * observation, including repeated assignments.</p>
 *
 * <p>Coefficient caps below constrain fitted energy corrections; they are
 * unrelated to the conditional mean-excess bound used for certification.
 * No final-sample observation is used here.</p>
 */
final class PackStarProposalLearning {
    private PackStarProposalLearning() {}

    static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double max = Math.max(a, b);
        return max + Math.log1p(Math.exp(Math.min(a, b) - max));
    }

    static final class Data {
        final double[] residual, logWeight;

        Data(int[][] conf, double[] residual, double[] logWeight) {
            if (conf.length != residual.length || conf.length != logWeight.length)
                throw new IllegalArgumentException("proposal sample shape mismatch");
            for (int i = 0; i < conf.length; i++) {
                if (conf[i] == null || !Double.isFinite(residual[i])
                        || !Double.isFinite(logWeight[i]))
                    throw new IllegalArgumentException("invalid proposal sample/source weight: " + i);
            }
            this.residual = residual;
            this.logWeight = logWeight;
        }
    }

    static final class JointFit {
        final double[] delta;
        final double offset;
        JointFit(double[] delta, double offset) { this.delta = delta; this.offset = offset; }
    }

    /** Weighted joint residual regression around the termwise prior.
     * Penalty per cell = its probability mass * shrink / distinct contexts.
     * Sparse cells and unseen cells therefore stay close to the existing model.
     */
    static JointFit jointFit(int[][] features, double[] residual, double[] logWeights,
                             double[] shrinkRatios, double cap, int sweeps) {
        int n = residual.length, p = shrinkRatios.length;
        if (n == 0 || features.length != n || logWeights.length != n
                || !(cap > 0) || !Double.isFinite(cap) || sweeps < 1)
            throw new IllegalArgumentException("invalid joint fit inputs");
        double logSum = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (!Double.isFinite(logWeights[i]) || !Double.isFinite(residual[i]))
                throw new IllegalArgumentException("nonfinite joint fit sample");
            logSum = logAdd(logSum, logWeights[i]);
        }
        List<List<Integer>> rows = new ArrayList<>(p);
        for (int j = 0; j < p; j++) {
            if (!(shrinkRatios[j] >= 0) || !Double.isFinite(shrinkRatios[j]))
                throw new IllegalArgumentException("invalid joint regularizer");
            rows.add(new ArrayList<>());
        }
        double[] weights = new double[n], errors = residual.clone(), delta = new double[p];
        double[] mass = new double[p];
        for (int i = 0; i < n; i++) {
            weights[i] = Math.exp(logWeights[i] - logSum);
            for (int j : features[i]) { rows.get(j).add(i); mass[j] += weights[i]; }
        }
        double offset = 0;
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double shift = 0, maxChange = 0;
            for (int i = 0; i < n; i++) shift += weights[i] * errors[i];
            offset += shift;
            for (int i = 0; i < n; i++) errors[i] -= shift;
            for (int j = 0; j < p; j++) {
                if (!(mass[j] > 0)) continue;
                double numerator = 0;
                for (int i : rows.get(j)) numerator += weights[i] * (errors[i] + delta[j]);
                double value = Math.max(-cap, Math.min(cap,
                        numerator / (mass[j] * (1 + shrinkRatios[j]))));
                double change = value - delta[j];
                delta[j] = value;
                for (int i : rows.get(j)) errors[i] -= change;
                maxChange = Math.max(maxChange, Math.abs(change));
            }
            if (maxChange < 1e-8 && Math.abs(shift) < 1e-8) break;
        }
        return new JointFit(delta, offset);
    }

    /** Anchor the tolerance to the global minimum, avoiding order-dependent
     * chains of pairwise 5% comparisons. The comparator orders complexity first.
     */
    static <T> T chooseWithinFivePercent(List<T> candidates,
            java.util.function.ToLongFunction<T> samples, Comparator<T> complexity) {
        long minimum = Long.MAX_VALUE;
        for (T candidate : candidates) minimum = Math.min(minimum, samples.applyAsLong(candidate));
        T best = null;
        for (T candidate : candidates) {
            if (samples.applyAsLong(candidate) > 1.05 * minimum + 1e-10) continue;
            if (best == null || complexity.compare(candidate, best) < 0) best = candidate;
        }
        return best;
    }

}
