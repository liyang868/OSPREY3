/*
** This file is part of OSPREY 3.0
**
** OSPREY Protein Redesign Software Version 3.0
** Copyright (C) 2001-2018 Bruce Donald Lab, Duke University
**
** OSPREY is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License version 2
** as published by the Free Software Foundation.
*/

package edu.duke.cs.osprey.packstar;

import org.apache.commons.math3.distribution.BetaDistribution;

import java.util.ArrayList;
import java.util.List;

/**
 * Fixed-batch PAC-style interval under a conditional mean-excess assumption.
 *
 * <p>Manuscript mapping: w is the importance weight, mu is an adaptation-only
 * scale, and kappa is the frozen relative clipping threshold. The clipped
 * component is y = min(w/(mu*kappa),1), the excess is
 * u = max(w/(mu*kappa)-1,0), and pi = P(u>0). The assumption is
 * E[u | u>0] <= m_u. Empirical Bernstein bounds E[y], while a one-sided
 * Clopper-Pearson bound piUpper gives E[u] <= m_u * piUpper.</p>
 *
 * <p>The caller supplies logRelativeWeights = log(w/mu) and logKappa =
 * log(kappa), then rescales the normalized endpoints by q_eta * mu * kappa.
 * The proposal, mu, kappa, m_u, sample size, and error allocation are frozen
 * before the independent final batch. This is not a confidence sequence.</p>
 *
 * <p>The historical class name is retained for compatibility. The bound m_u
 * is an assumption on the conditional mean, not a pointwise bound on u.
 * Observed mean/max excess values are diagnostics, not replacements for m_u.</p>
 */
public final class PackStarFrequencySeverityPAC {

    private PackStarFrequencySeverityPAC() {}

    /** Point estimates used only for adaptation-time sizing and ranking. */
    public static final class Moments {
        public final double clippedMean;
        public final double clippedVariance;
        public final double exceedanceProbability;

        public Moments(double clippedMean, double clippedVariance,
                       double exceedanceProbability) {
            requireProbability(clippedMean, "clipped-component mean");
            if (!Double.isFinite(clippedVariance) || clippedVariance < 0.0) {
                throw new IllegalArgumentException(
                        "clipped-component variance must be finite and nonnegative");
            }
            requireProbability(exceedanceProbability, "tail probability");
            this.clippedMean = clippedMean;
            this.clippedVariance = clippedVariance;
            this.exceedanceProbability = exceedanceProbability;
        }
    }

    /**
     * Aggregated cross-fit diagnostics for fold-specific, self-normalized
     * proposal scores.  The folds may use different fitted eta values, so this
     * object describes their sample-size-weighted mixture and is adaptation
     * evidence only; it is never an issued confidence interval.
     */
    static final class PooledCrossfitMoments {
        final int sampleCount;
        final double effectiveSampleSize;
        final double effectiveSampleFraction;
        final Moments moments;

        private PooledCrossfitMoments(int sampleCount,
                                      double effectiveSampleSize,
                                      Moments moments) {
            this.sampleCount = sampleCount;
            this.effectiveSampleSize = effectiveSampleSize;
            this.effectiveSampleFraction =
                    effectiveSampleSize / sampleCount;
            this.moments = moments;
        }
    }

    /**
     * Pool normalized weighted moments from disjoint cross-fit folds.
     *
     * <p>Within fold {@code f}, weights sum to one and have
     * {@code sum(w^2)=1/ESS_f}.  Giving the fold mass {@code n_f/N} produces
     * exact pooled weighted mean, variance, and ESS identities without
     * pretending that either fold-specific eta was fitted on its validation
     * observations.</p>
     */
    static PooledCrossfitMoments poolCrossfitMoments(
            int[] sampleCounts,
            double[] effectiveSampleSizes,
            double[] clippedMeans,
            double[] clippedVariances,
            double[] exceedanceProbabilities) {
        if (sampleCounts == null || sampleCounts.length == 0
                || effectiveSampleSizes == null
                || clippedMeans == null || clippedVariances == null
                || exceedanceProbabilities == null
                || effectiveSampleSizes.length != sampleCounts.length
                || clippedMeans.length != sampleCounts.length
                || clippedVariances.length != sampleCounts.length
                || exceedanceProbabilities.length != sampleCounts.length) {
            throw new IllegalArgumentException(
                    "cross-fit moment arrays must have one common positive length");
        }

        int total = 0;
        for (int count : sampleCounts) {
            if (count <= 1 || total > Integer.MAX_VALUE - count) {
                throw new IllegalArgumentException(
                        "cross-fit fold sample counts must exceed one and have a finite sum");
            }
            total += count;
        }

        double pooledMean = 0.0;
        double pooledSecond = 0.0;
        double pooledTail = 0.0;
        double pooledWeightSquares = 0.0;
        for (int fold = 0; fold < sampleCounts.length; fold++) {
            int count = sampleCounts[fold];
            double ess = effectiveSampleSizes[fold];
            double mean = clippedMeans[fold];
            double variance = clippedVariances[fold];
            double tail = exceedanceProbabilities[fold];
            if (!Double.isFinite(ess) || !(ess > 0.0)
                    || ess > count * (1.0 + 1.0e-9)
                    || !Double.isFinite(mean) || mean < 0.0 || mean > 1.0
                    || !Double.isFinite(variance) || variance < 0.0
                    || !Double.isFinite(tail) || tail < 0.0 || tail > 1.0) {
                throw new IllegalArgumentException(
                        "invalid cross-fit weighted fold moments");
            }
            ess = Math.min(ess, count);
            double foldMass = (double) count / total;
            double foldWeightSquares = 1.0 / ess;
            double centeredNumerator =
                    variance * Math.max(0.0, 1.0 - foldWeightSquares);
            pooledMean += foldMass * mean;
            pooledSecond += foldMass
                    * (centeredNumerator + mean * mean);
            pooledTail += foldMass * tail;
            pooledWeightSquares += foldMass * foldMass
                    * foldWeightSquares;
        }
        if (!(pooledWeightSquares > 0.0)
                || !(pooledWeightSquares < 1.0)
                || !Double.isFinite(pooledWeightSquares)) {
            throw new IllegalArgumentException(
                    "cross-fit pooled weights have invalid concentration");
        }
        double varianceNumerator = Math.max(0.0,
                pooledSecond - pooledMean * pooledMean);
        double pooledVariance = varianceNumerator
                / (1.0 - pooledWeightSquares);
        Moments moments = new Moments(
                Math.max(0.0, Math.min(1.0, pooledMean)),
                pooledVariance,
                Math.max(0.0, Math.min(1.0, pooledTail)));
        return new PooledCrossfitMoments(
                total, 1.0 / pooledWeightSquares, moments);
    }

    /** A finite-sample interval in units of the frozen clipping scale. */
    public static final class Interval {
        public final int sampleCount;
        public final int exceedanceCount;
        public final double clippedMean;
        public final double clippedVariance;
        public final double clippedRadius;
        public final double clippedLower;
        public final double clippedUpper;
        public final double exceedanceProbabilityEmpirical;
        public final double exceedanceProbabilityUpper;
        public final double conditionalMeanExcessBound;
        public final double excessMeanUpper;
        public final double normalizedMeanLower;
        public final double normalizedMeanUpper;
        public final double epsilon;
        public final double empiricalExcessMean;
        public final double empiricalConditionalMeanExcess;
        public final double observedMaxExcess;

        private Interval(int sampleCount, int exceedanceCount,
                         double clippedMean, double clippedVariance,
                         double clippedRadius, double clippedLower,
                         double clippedUpper,
                         double exceedanceProbabilityEmpirical,
                         double exceedanceProbabilityUpper,
                         double conditionalMeanExcessBound,
                         double excessMeanUpper,
                         double normalizedMeanLower,
                         double normalizedMeanUpper,
                         double epsilon,
                         double empiricalExcessMean,
                         double empiricalConditionalMeanExcess,
                         double observedMaxExcess) {
            this.sampleCount = sampleCount;
            this.exceedanceCount = exceedanceCount;
            this.clippedMean = clippedMean;
            this.clippedVariance = clippedVariance;
            this.clippedRadius = clippedRadius;
            this.clippedLower = clippedLower;
            this.clippedUpper = clippedUpper;
            this.exceedanceProbabilityEmpirical = exceedanceProbabilityEmpirical;
            this.exceedanceProbabilityUpper = exceedanceProbabilityUpper;
            this.conditionalMeanExcessBound = conditionalMeanExcessBound;
            this.excessMeanUpper = excessMeanUpper;
            this.normalizedMeanLower = normalizedMeanLower;
            this.normalizedMeanUpper = normalizedMeanUpper;
            this.epsilon = epsilon;
            this.empiricalExcessMean = empiricalExcessMean;
            this.empiricalConditionalMeanExcess =
                    empiricalConditionalMeanExcess;
            this.observedMaxExcess =
                    observedMaxExcess;
        }

        public boolean hasPositiveBulkLower() {
            return clippedLower > 0.0;
        }

        public boolean isFinite() {
            return Double.isFinite(normalizedMeanUpper)
                    && normalizedMeanUpper > 0.0
                    && normalizedMeanLower >= 0.0
                    && normalizedMeanLower <= normalizedMeanUpper
                    && Double.isFinite(epsilon);
        }

        public Moments moments() {
            return new Moments(clippedMean, clippedVariance,
                    exceedanceProbabilityEmpirical);
        }
    }

    /** Adaptation-only projected sample size.  It is not a certificate. */
    public static final class Sizing {
        public final int finalSamples;
        public final boolean reachableAtMax;
        public final double epsilonAtFinalSamples;
        public final double epsilonAtMaxSamples;
        public final double sizingTarget;

        private Sizing(int finalSamples, boolean reachableAtMax,
                       double epsilonAtFinalSamples,
                       double epsilonAtMaxSamples,
                       double sizingTarget) {
            this.finalSamples = finalSamples;
            this.reachableAtMax = reachableAtMax;
            this.epsilonAtFinalSamples = epsilonAtFinalSamples;
            this.epsilonAtMaxSamples = epsilonAtMaxSamples;
            this.sizingTarget = sizingTarget;
        }
    }

    /** Fixed-batch test result that can reject, but never validate, m_u. */
    public static final class MeanExcessTest {
        public final int exceedanceCount;
        public final boolean sufficientTailSamples;
        public final boolean logicalViolation;
        public final boolean rejected;
        public final double logPValue;
        public final double pValueUpper;

        private MeanExcessTest(int exceedanceCount,
                             boolean sufficientTailSamples,
                             boolean logicalViolation,
                             boolean rejected,
                             double logPValue,
                             double pValueUpper) {
            this.exceedanceCount = exceedanceCount;
            this.sufficientTailSamples = sufficientTailSamples;
            this.logicalViolation = logicalViolation;
            this.rejected = rejected;
            this.logPValue = logPValue;
            this.pValueUpper = pValueUpper;
        }
    }

    /** Evaluate a fresh IID sample of log relative weights. */
    public static Interval evaluate(double[] logRelativeWeights,
                                    double logKappa,
                                    double conditionalMeanExcessBound,
                                    double clippedComponentDelta,
                                    double exceedanceProbabilityDelta) {
        requireLogWeights(logRelativeWeights);
        if (!Double.isFinite(logKappa)) {
            throw new IllegalArgumentException(
                    "relative log clip must be finite");
        }
        requireMeanExcessBound(conditionalMeanExcessBound);
        requireDelta(clippedComponentDelta, "clipped-component delta");
        requireDelta(exceedanceProbabilityDelta, "exceedance-probability delta");

        int n = logRelativeWeights.length;
        double[] clippedSamples = new double[n];
        int exceedanceCount = 0;
        double sumTail = 0.0;
        double maxTail = 0.0;
        boolean infiniteTail = false;

        for (int i = 0; i < n; i++) {
            double relative = logRelativeWeights[i] - logKappa;
            if (relative > 0.0) {
                clippedSamples[i] = 1.0;
                exceedanceCount++;
                double excess = safeExpm1(relative);
                if (Double.isFinite(excess) && !infiniteTail) {
                    sumTail += excess;
                    if (!Double.isFinite(sumTail)) infiniteTail = true;
                } else {
                    infiniteTail = true;
                }
                maxTail = Math.max(maxTail, excess);
            } else {
                clippedSamples[i] = Math.exp(relative);
            }
        }

        double mean = mean(clippedSamples);
        double variance = sampleVariance(clippedSamples, mean);
        double radius = empiricalBernsteinRadius(
                n, variance, 1.0, clippedComponentDelta);
        double lower = Math.max(0.0, mean - radius);
        double upper = Math.min(1.0, mean + radius);
        double pHat = (double) exceedanceCount / n;
        double pUpper = clopperPearsonUpper(
                exceedanceCount, n, exceedanceProbabilityDelta);
        double tailUpper = pUpper * conditionalMeanExcessBound;
        double meanUpper = upper + tailUpper;
        double epsilon = intervalEpsilon(lower, meanUpper);
        double empiricalExcessMean = infiniteTail
                ? Double.POSITIVE_INFINITY : sumTail / n;
        double empiricalConditionalMeanExcess = exceedanceCount > 0
                ? empiricalExcessMean / pHat : 0.0;

        return new Interval(
                n, exceedanceCount, mean, variance, radius, lower, upper,
                pHat, pUpper, conditionalMeanExcessBound, tailUpper,
                lower, meanUpper, epsilon, empiricalExcessMean,
                empiricalConditionalMeanExcess, maxTail);
    }

    /**
     * Project an interval from pilot/cross-fit point estimates.  This is a
     * design diagnostic only; only {@link #evaluate} on fresh IID samples is
     * the issued confidence interval.
     */
    public static Interval project(int sampleCount, Moments moments,
                                   double conditionalMeanExcessBound,
                                   double clippedComponentDelta,
                                   double exceedanceProbabilityDelta) {
        if (sampleCount <= 1) {
            throw new IllegalArgumentException(
                    "projected sample count must exceed one");
        }
        if (moments == null) {
            throw new IllegalArgumentException("moments are required");
        }
        requireMeanExcessBound(conditionalMeanExcessBound);
        requireDelta(clippedComponentDelta, "clipped-component delta");
        requireDelta(exceedanceProbabilityDelta, "exceedance-probability delta");

        double radius = empiricalBernsteinRadius(
                sampleCount, moments.clippedVariance, 1.0, clippedComponentDelta);
        double lower = Math.max(0.0, moments.clippedMean - radius);
        double upper = Math.min(1.0, moments.clippedMean + radius);
        int pseudoExceedanceCount = (int) Math.min(sampleCount,
                Math.ceil(moments.exceedanceProbability * sampleCount));
        double pUpper = clopperPearsonUpper(
                pseudoExceedanceCount, sampleCount, exceedanceProbabilityDelta);
        double tailUpper = pUpper * conditionalMeanExcessBound;
        double meanUpper = upper + tailUpper;
        double epsilon = intervalEpsilon(lower, meanUpper);

        return new Interval(
                sampleCount, pseudoExceedanceCount, moments.clippedMean,
                moments.clippedVariance, radius, lower, upper,
                moments.exceedanceProbability, pUpper,
                conditionalMeanExcessBound, tailUpper,
                lower, meanUpper, epsilon,
                Double.NaN, Double.NaN, Double.NaN);
    }

    /** Choose a pilot-frozen final N, including the unreachable diagnostic cap. */
    public static Sizing size(Moments moments,
                              int maxSamples,
                              int unreachableSamples,
                              double targetEpsilon,
                              double safetyFraction,
                              double conditionalMeanExcessBound,
                              double clippedComponentDelta,
                              double exceedanceProbabilityDelta) {
        validateSizingInputs(moments, maxSamples, unreachableSamples,
                targetEpsilon, safetyFraction, conditionalMeanExcessBound,
                clippedComponentDelta, exceedanceProbabilityDelta);

        double sizingTarget = targetEpsilon * safetyFraction;
        Interval atMax = project(maxSamples, moments,
                conditionalMeanExcessBound, clippedComponentDelta, exceedanceProbabilityDelta);
        if (atMax.epsilon > targetEpsilon) {
            Interval unreachable = project(unreachableSamples, moments,
                    conditionalMeanExcessBound, clippedComponentDelta, exceedanceProbabilityDelta);
            return new Sizing(unreachableSamples, false,
                    unreachable.epsilon, atMax.epsilon, sizingTarget);
        }
        if (atMax.epsilon > sizingTarget) {
            return new Sizing(maxSamples, true, atMax.epsilon,
                    atMax.epsilon, sizingTarget);
        }

        int low = 2;
        int high = maxSamples;
        while (low < high) {
            int middle = (low + high) >>> 1;
            double epsilon = project(middle, moments,
                    conditionalMeanExcessBound, clippedComponentDelta,
                    exceedanceProbabilityDelta).epsilon;
            if (epsilon <= sizingTarget) high = middle;
            else low = middle + 1;
        }
        Interval selected = project(low, moments,
                conditionalMeanExcessBound, clippedComponentDelta, exceedanceProbabilityDelta);
        return new Sizing(low, true, selected.epsilon,
                atMax.epsilon, sizingTarget);
    }

    /** One-sided exact Clopper-Pearson upper confidence limit. */
    public static double clopperPearsonUpper(int successes, int trials,
                                             double delta) {
        if (trials <= 0 || successes < 0 || successes > trials) {
            throw new IllegalArgumentException(
                    "invalid binomial successes/trials");
        }
        requireDelta(delta, "binomial delta");
        if (successes == trials) return 1.0;
        if (successes == 0) {
            return -Math.expm1(Math.log(delta) / trials);
        }
        BetaDistribution beta = new BetaDistribution(
                successes + 1.0, trials - successes);
        double value = beta.inverseCumulativeProbability(1.0 - delta);
        return Math.max(0.0, Math.min(1.0, value));
    }

    /** Two-sided Maurer-Pontil radius for values in an interval of this range. */
    public static double empiricalBernsteinRadius(int sampleCount,
                                                  double sampleVariance,
                                                  double range,
                                                  double delta) {
        if (sampleCount <= 1) return Double.POSITIVE_INFINITY;
        if (!Double.isFinite(sampleVariance) || sampleVariance < 0.0
                || !Double.isFinite(range) || range < 0.0) {
            throw new IllegalArgumentException(
                    "invalid variance/range for empirical Bernstein");
        }
        requireDelta(delta, "empirical-Bernstein delta");
        double logTerm = Math.log(4.0 / delta);
        return Math.sqrt(2.0 * sampleVariance * logTerm / sampleCount)
                + 7.0 * range * logTerm
                / (3.0 * (sampleCount - 1));
    }

    /**
     * Test the conditional mean-excess assumption on a fresh, fixed-size IID batch.
     * The proposal, clipping scale, cap, and sample count must be fixed before
     * drawing it. The same batch may also supply the partition-function
     * interval: rejection merely suppresses an otherwise reported estimate.
     * Failure to reject is never interpreted as validation. Markov
     * inequality gives p=min(1,m_u/meanPositiveExcess), conditional on the
     * number of exceedances. This is a fixed-batch, not sequential, test.
     */
    public static MeanExcessTest testConditionalMeanExcess(
            double[] logRelativeWeights,
            double logKappa,
            double conditionalMeanExcessBound,
            double testAlpha) {
        requireLogWeights(logRelativeWeights);
        if (!Double.isFinite(logKappa)) {
            throw new IllegalArgumentException(
                    "relative log clip must be finite");
        }
        requireMeanExcessBound(conditionalMeanExcessBound);
        requireDelta(testAlpha, "mean-excess diagnostic alpha");

        List<Double> logExcesses = new ArrayList<>();
        for (double logWeight : logRelativeWeights) {
            double relative = logWeight - logKappa;
            if (relative <= 0.0) continue;
            if (relative > 40.0) {
                logExcesses.add(relative + Math.log1p(-Math.exp(-relative)));
            } else {
                logExcesses.add(Math.log(Math.expm1(relative)));
            }
        }
        int exceedanceCount = logExcesses.size();
        if (exceedanceCount == 0) {
            return new MeanExcessTest(0, false, false, false, 0.0, 1.0);
        }
        if (conditionalMeanExcessBound == 0.0) {
            return new MeanExcessTest(exceedanceCount, true, true, true,
                    Double.NEGATIVE_INFINITY, 0.0);
        }
        // Compute in log space, including a single exceedance and huge tails.
        double logMeanExcess = logSumExp(logExcesses) - Math.log(exceedanceCount);
        double logP = Math.min(0.0, Math.log(conditionalMeanExcessBound) - logMeanExcess);
        double p = Math.exp(logP);
        boolean rejected = logP <= Math.log(testAlpha);
        return new MeanExcessTest(exceedanceCount, true, false, rejected, logP, p);
    }

    private static double intervalEpsilon(double lower, double upper) {
        if (!(upper > 0.0) || !Double.isFinite(upper)
                || lower < 0.0 || lower > upper) return 1.0;
        return Math.max(0.0, Math.min(1.0, 1.0 - lower / upper));
    }

    private static double mean(double[] values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / values.length;
    }

    private static double sampleVariance(double[] values, double mean) {
        double sum = 0.0;
        for (double value : values) {
            double delta = value - mean;
            sum += delta * delta;
        }
        return Math.max(0.0, sum / (values.length - 1));
    }

    private static double safeExpm1(double value) {
        if (value > Math.log(Double.MAX_VALUE)) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.expm1(value);
    }

    private static double logSumExp(List<Double> values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) max = Math.max(max, value);
        if (max == Double.POSITIVE_INFINITY) return max;
        if (max == Double.NEGATIVE_INFINITY) return max;
        double sum = 0.0;
        for (double value : values) sum += Math.exp(value - max);
        return max + Math.log(sum);
    }

    private static void requireLogWeights(double[] logWeights) {
        if (logWeights == null || logWeights.length <= 1) {
            throw new IllegalArgumentException(
                    "at least two log relative weights are required");
        }
        for (double value : logWeights) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "log relative weights must be finite");
            }
        }
    }

    /** Validate sizing inputs without projecting intervals or searching for a sample count. */
    static void validateSizingInputs(Moments moments,
                                     int maxSamples,
                                     int unreachableSamples,
                                     double targetEpsilon,
                                     double safetyFraction,
                                     double conditionalMeanExcessBound,
                                     double clippedComponentDelta,
                                     double exceedanceProbabilityDelta) {
        if (maxSamples <= 1 || unreachableSamples <= 1
                || unreachableSamples > maxSamples) {
            throw new IllegalArgumentException(
                    "invalid mean-excess maximum/unreachable sample counts");
        }
        if (!Double.isFinite(targetEpsilon)
                || !(targetEpsilon > 0.0) || targetEpsilon >= 1.0) {
            throw new IllegalArgumentException(
                    "target epsilon must be finite and in (0,1)");
        }
        if (!Double.isFinite(safetyFraction)
                || !(safetyFraction > 0.0) || safetyFraction > 1.0) {
            throw new IllegalArgumentException(
                    "sizing safety fraction must be in (0,1]");
        }
        if (moments == null) {
            throw new IllegalArgumentException("moments are required");
        }
        requireMeanExcessBound(conditionalMeanExcessBound);
        requireDelta(clippedComponentDelta, "clipped-component delta");
        requireDelta(exceedanceProbabilityDelta, "exceedance-probability delta");
    }

    private static void requireMeanExcessBound(double meanExcessBound) {
        if (!Double.isFinite(meanExcessBound) || meanExcessBound < 0.0) {
            throw new IllegalArgumentException(
                    "conditional mean-excess bound must be finite and nonnegative");
        }
    }

    private static void requireDelta(double delta, String label) {
        if (!Double.isFinite(delta) || !(delta > 0.0) || delta >= 1.0) {
            throw new IllegalArgumentException(label + " must be in (0,1)");
        }
    }

    private static void requireProbability(double value, String label) {
        if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException(label + " must be in [0,1]");
        }
    }
}
