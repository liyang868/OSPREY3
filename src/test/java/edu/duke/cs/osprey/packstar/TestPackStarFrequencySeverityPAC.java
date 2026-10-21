package edu.duke.cs.osprey.packstar;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class TestPackStarFrequencySeverityPAC {

    @Test
    public void clopperPearsonZeroCountUsesExactClosedForm() {
        int n = 100;
        double delta = 0.025;
        double expected = -Math.expm1(Math.log(delta) / n);
        assertEquals(expected,
                PackStarFrequencySeverityPAC.clopperPearsonUpper(
                        0, n, delta), 1.0e-15);
    }

    @Test
    public void meanExcessIntervalMatchesItsDecomposition() {
        double[] logRelative = {
                Math.log(1.0), Math.log(2.0),
                Math.log(4.0), Math.log(0.5)
        };
        double logClip = Math.log(2.0);
        double severity = 3.0;
        double clippedComponentDelta = 0.025;
        double exceedanceProbabilityDelta = 0.025;

        PackStarFrequencySeverityPAC.Interval interval =
                PackStarFrequencySeverityPAC.evaluate(
                        logRelative, logClip, severity,
                        clippedComponentDelta, exceedanceProbabilityDelta);

        // Y=min(R/C,1) = [0.5,1,1,0.25].  Only R=4 is a strict tail.
        assertEquals(4, interval.sampleCount);
        assertEquals(1, interval.exceedanceCount);
        assertEquals(0.6875, interval.clippedMean, 0.0);
        assertEquals(0.25, interval.exceedanceProbabilityEmpirical, 0.0);
        assertEquals(0.25, interval.empiricalExcessMean, 1.0e-15);
        assertEquals(1.0, interval.empiricalConditionalMeanExcess, 1.0e-15);
        assertEquals(1.0, interval.observedMaxExcess, 1.0e-15);
        assertEquals(interval.exceedanceProbabilityUpper * severity,
                interval.excessMeanUpper, 0.0);
        assertEquals(interval.clippedUpper + interval.excessMeanUpper,
                interval.normalizedMeanUpper, 0.0);
        assertEquals(1.0 - interval.normalizedMeanLower
                        / interval.normalizedMeanUpper,
                interval.epsilon, 1.0e-15);
    }

    @Test
    public void addingARawEnergyGaugeDoesNotChangeRelativeInterval() {
        double[] rawLogWeights = {-1.5, -0.2, 0.1, 1.7, 2.2};
        double logMu = 0.3;
        double gauge = 19.0;
        double[] relativeBefore = new double[rawLogWeights.length];
        double[] relativeAfter = new double[rawLogWeights.length];
        for (int i = 0; i < rawLogWeights.length; i++) {
            relativeBefore[i] = rawLogWeights[i] - logMu;
            relativeAfter[i] = rawLogWeights[i] + gauge
                    - (logMu + gauge);
        }
        PackStarFrequencySeverityPAC.Interval before =
                PackStarFrequencySeverityPAC.evaluate(
                        relativeBefore, 1.0, 20.0, 0.025, 0.025);
        PackStarFrequencySeverityPAC.Interval after =
                PackStarFrequencySeverityPAC.evaluate(
                        relativeAfter, 1.0, 20.0, 0.025, 0.025);
        assertEquals(before.exceedanceCount, after.exceedanceCount);
        assertEquals(before.clippedMean, after.clippedMean, 1.0e-15);
        assertEquals(before.clippedLower, after.clippedLower, 1.0e-15);
        assertEquals(before.normalizedMeanUpper,
                after.normalizedMeanUpper, 1.0e-15);
        assertEquals(before.epsilon, after.epsilon, 1.0e-15);
    }

    @Test
    public void pooledCrossfitMomentsRecoverUniformCombinedSample() {
        PackStarFrequencySeverityPAC.PooledCrossfitMoments pooled =
                PackStarFrequencySeverityPAC.poolCrossfitMoments(
                        new int[]{2, 2},
                        new double[]{2.0, 2.0},
                        new double[]{0.5, 0.5},
                        new double[]{0.5, 0.5},
                        new double[]{0.0, 1.0});
        assertEquals(4, pooled.sampleCount);
        assertEquals(4.0, pooled.effectiveSampleSize, 1.0e-15);
        assertEquals(1.0, pooled.effectiveSampleFraction, 1.0e-15);
        assertEquals(0.5, pooled.moments.clippedMean, 1.0e-15);
        assertEquals(1.0 / 3.0,
                pooled.moments.clippedVariance, 1.0e-15);
        assertEquals(0.5, pooled.moments.exceedanceProbability, 1.0e-15);
    }

    @Test
    public void pooledCrossfitEssAccountsForFoldWeightConcentration() {
        PackStarFrequencySeverityPAC.PooledCrossfitMoments pooled =
                PackStarFrequencySeverityPAC.poolCrossfitMoments(
                        new int[]{100, 100},
                        new double[]{25.0, 100.0},
                        new double[]{0.4, 0.6},
                        new double[]{0.02, 0.02},
                        new double[]{0.1, 0.3});
        assertEquals(80.0, pooled.effectiveSampleSize, 1.0e-12);
        assertEquals(0.4, pooled.effectiveSampleFraction, 1.0e-15);
        assertEquals(0.5, pooled.moments.clippedMean, 1.0e-15);
        assertEquals(0.2, pooled.moments.exceedanceProbability, 1.0e-15);
        assertThrows(IllegalArgumentException.class,
                () -> PackStarFrequencySeverityPAC.poolCrossfitMoments(
                        new int[]{10}, new double[]{11.0},
                        new double[]{0.5}, new double[]{0.1},
                        new double[]{0.0}));
    }

    @Test
    public void sizingUsesTheUnreachableCapWhenTargetCannotBeReached() {
        PackStarFrequencySeverityPAC.Moments collapsed =
                new PackStarFrequencySeverityPAC.Moments(
                        0.01, 0.2, 0.5);
        PackStarFrequencySeverityPAC.Sizing sizing =
                PackStarFrequencySeverityPAC.size(
                        collapsed, 4000, 400,
                        0.683, 0.9, 20.0,
                        0.025, 0.025);
        assertFalse(sizing.reachableAtMax);
        assertEquals(400, sizing.finalSamples);
        assertTrue(sizing.epsilonAtMaxSamples > 0.683);
    }

    @Test
    public void sizingFindsAReachableFrozenCount() {
        PackStarFrequencySeverityPAC.Moments stable =
                new PackStarFrequencySeverityPAC.Moments(
                        0.8, 0.01, 0.0);
        PackStarFrequencySeverityPAC.Sizing sizing =
                PackStarFrequencySeverityPAC.size(
                        stable, 4000, 400,
                        0.683, 0.9, 1.0,
                        0.025, 0.025);
        assertTrue(sizing.reachableAtMax);
        assertTrue(sizing.finalSamples >= 2);
        assertTrue(sizing.finalSamples <= 4000);
        assertTrue(sizing.epsilonAtFinalSamples
                <= 0.683 * 0.9 + 1.0e-12);
    }

    @Test
    public void freshSeverityBatchCanRejectButNotValidate() {
        double[] hugeTail = new double[10];
        for (int i = 0; i < hugeTail.length; i++) {
            hugeTail[i] = Math.log1p(100.0);
        }
        PackStarFrequencySeverityPAC.MeanExcessTest rejected =
                PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        hugeTail, 0.0, 1.0, 0.05);
        assertTrue(rejected.sufficientTailSamples);
        assertTrue(rejected.rejected);
        assertTrue(rejected.pValueUpper <= 0.05);

        double[] noTail = {-2.0, -1.0, -0.5};
        PackStarFrequencySeverityPAC.MeanExcessTest inconclusive =
                PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        noTail, 0.0, 1.0, 0.05);
        assertFalse(inconclusive.sufficientTailSamples);
        assertFalse(inconclusive.rejected);
    }

    @Test
    public void finalBatchSuppliesBothIntervalAndSeverityTest() {
        double[] finalWeights = new double[4000];
        Arrays.fill(finalWeights, Math.log(0.5));
        Arrays.fill(finalWeights, 0, 13, Math.log1p(1445.0));
        PackStarFrequencySeverityPAC.Interval interval =
                PackStarFrequencySeverityPAC.evaluate(
                        finalWeights, 0.0, 20.0, 0.025, 0.025);
        PackStarFrequencySeverityPAC.MeanExcessTest test =
                PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        finalWeights, 0.0, 20.0, 0.05);
        assertEquals(13, interval.exceedanceCount);
        assertEquals(interval.exceedanceCount, test.exceedanceCount);
        assertTrue(test.rejected);
        assertTrue(test.pValueUpper <= 0.05);
    }

    @Test
    public void oneLargeTailAmongSmallTailsDoesNotImplyRejection() {
        double[] finalWeights = new double[4000];
        Arrays.fill(finalWeights, Math.log(0.5));
        Arrays.fill(finalWeights, 0, 12, Math.log1p(0.1));
        finalWeights[12] = Math.log1p(3875.0);
        PackStarFrequencySeverityPAC.Interval interval =
                PackStarFrequencySeverityPAC.evaluate(
                        finalWeights, 0.0, 20.0, 0.025, 0.025);
        PackStarFrequencySeverityPAC.MeanExcessTest test =
                PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        finalWeights, 0.0, 20.0, 0.05);
        assertTrue(interval.empiricalConditionalMeanExcess > 200.0);
        assertTrue(test.sufficientTailSamples);
        assertFalse(test.rejected);
    }

    @Test
    public void singleTailCanRejectPositiveOrZeroCap() {
        double[] weights = {Math.log(0.5), Math.log1p(1.0e9)};
        PackStarFrequencySeverityPAC.MeanExcessTest positiveCap =
                PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        weights, 0.0, 20.0, 0.05);
        assertEquals(1, positiveCap.exceedanceCount);
        assertTrue(positiveCap.sufficientTailSamples);
        assertTrue(positiveCap.rejected);
        assertEquals(2.0e-8, positiveCap.pValueUpper, 1.0e-18);
        PackStarFrequencySeverityPAC.MeanExcessTest zeroCap =
                PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        weights, 0.0, 0.0, 0.05);
        assertTrue(zeroCap.logicalViolation);
        assertTrue(zeroCap.rejected);
    }

    @Test
    public void fixedBatchPvalueIsSuperUniformWithRandomTailCount() {
        // Probabilities .5/.49/.01 for U=0/1/951 give E[U|U>0]=20.
        // Exhaust all 3^4 outcomes, including zero and one exceedance.
        for (double alpha : new double[]{0.01, 0.05, 0.10, 0.5}) {
            double rejectionProbability = 0.0;
            for (int code = 0; code < 81; code++) {
                double[] weights = new double[4];
                double probability = 1.0;
                int digits = code;
                for (int i = 0; i < 4; i++, digits /= 3) {
                    int value = digits % 3;
                    weights[i] = value == 0 ? Math.log(0.5)
                            : Math.log1p(value == 1 ? 1.0 : 951.0);
                    probability *= value == 0 ? .5 : value == 1 ? .49 : .01;
                }
                var test = PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                        weights, 0.0, 20.0, alpha);
                if (test.rejected) rejectionProbability += probability;
            }
            assertTrue(rejectionProbability <= alpha + 1e-12);
        }
    }

    @Test
    public void meanPvalueDistinguishesFiveAndTenPercent() {
        double[] weights = {Math.log(0.5), Math.log1p(298.155882256389)};
        var five = PackStarFrequencySeverityPAC.testConditionalMeanExcess(weights, 0, 20, .05);
        var ten = PackStarFrequencySeverityPAC.testConditionalMeanExcess(weights, 0, 20, .10);
        assertFalse(five.rejected);
        assertTrue(ten.rejected);
        assertEquals(0.06707900527953253, ten.pValueUpper, 1e-14);
    }

    @Test
    public void meanPvalueHandlesNoTailAndOverflow() {
        var none = PackStarFrequencySeverityPAC.testConditionalMeanExcess(new double[]{-1, 0}, 0, 20, .10);
        assertEquals(1.0, none.pValueUpper, 0.0);
        assertFalse(none.sufficientTailSamples);
        var huge = PackStarFrequencySeverityPAC.testConditionalMeanExcess(new double[]{-1, 1000}, 0, 20, .10);
        assertTrue(huge.rejected);
        assertTrue(Double.isFinite(huge.logPValue));
        assertEquals(0.0, huge.pValueUpper, 0.0);
        var mild = PackStarFrequencySeverityPAC.testConditionalMeanExcess(new double[]{-1, Math.log1p(61.83)}, 0, 20, .10);
        assertFalse(mild.rejected);
    }

    @Test
    public void productionShrinkGridIncludesTheUnshrunkCandidate() {
        PackStarEstimator.MeanExcessShrinkPair[] shrink =
                PackStarEstimator.parseFrequencySeverityShrinkGrid(
                        "0:0,2:5,5:10");
        assertEquals(3, shrink.length);
        assertEquals(0.0, shrink[0].unary, 0.0);
        assertEquals(0.0, shrink[0].pair, 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> PackStarEstimator.parseFrequencySeverityShrinkGrid(
                        "2:5"));
    }

    @Test
    public void candidateIdentityRecordsTheFixedPairCorrectionScale() {
        PackStarEstimator.MeanExcessShrinkPair raw =
                new PackStarEstimator.MeanExcessShrinkPair(0.0, 0.0);
        assertEquals("ku-0.00000-kp-0.00000-alpha-1.00000-pair-only",
                PackStarEstimator.meanExcessCandidateId(raw));
    }

    @Test
    public void sourceAwareWeightRetainsTheSourceNormalizer() {
        double rt = 0.6;
        double targetEnergy = 13.0;
        double sourceEnergy = 11.5;
        double sourceLogZ = 27.0;
        assertEquals(-(targetEnergy - sourceEnergy) / rt + sourceLogZ,
                PackStarEstimator.meanExcessSourceLogWeight(
                        targetEnergy, sourceEnergy, sourceLogZ, rt),
                0.0);
    }

    @Test
    public void sourceAwareWeightIsInvariantToASourceEnergyGauge() {
        double rt = 0.6;
        double targetEnergy = 13.0;
        double sourceEnergy = 11.5;
        double sourceLogZ = 27.0;
        double gauge = 4.25;
        double before = PackStarEstimator.meanExcessSourceLogWeight(
                targetEnergy, sourceEnergy, sourceLogZ, rt);
        // Adding g to every source energy changes logZ_source by -g/RT.
        double after = PackStarEstimator.meanExcessSourceLogWeight(
                targetEnergy, sourceEnergy + gauge,
                sourceLogZ - gauge / rt, rt);
        assertEquals(before, after, 1.0e-14);
    }

    @Test
    public void zeroBulkLowerWithReachablePointSizingExtendsFrozenDiscovery() {
        double logClip = 1.0;
        double[] logRelative = new double[100];
        for (int i = 0; i < logRelative.length; i++) {
            // Twenty observations at the clipping boundary and eighty nearly
            // zero clipped component values mimic a high-variance, tail-free pilot.
            logRelative[i] = i < 20
                    ? logClip : logClip + Math.log(1.0e-12);
        }
        PackStarFrequencySeverityPAC.Interval interval =
                PackStarFrequencySeverityPAC.evaluate(
                        logRelative, logClip, 20.0, 0.025, 0.025);
        PackStarFrequencySeverityPAC.Sizing sizing =
                PackStarFrequencySeverityPAC.size(
                        new PackStarFrequencySeverityPAC.Moments(
                                interval.clippedMean,
                                interval.clippedVariance * 1.3,
                                interval.exceedanceProbabilityEmpirical),
                        4000, 400, 0.683, 0.9, 20.0,
                        0.025, 0.025);

        assertFalse(interval.hasPositiveBulkLower());
        assertTrue(sizing.reachableAtMax);
        assertTrue(PackStarEstimator.shouldExtendFrequencySeverityDiscovery(
                100, 400, interval, sizing, 0.683));
        assertFalse(PackStarEstimator.shouldExtendFrequencySeverityDiscovery(
                400, 400, interval, sizing, 0.683));
    }

    @Test
    public void discoveryExtensionDoesNotMaskAnUnreachableProposal() {
        double[] logRelative = new double[100];
        Arrays.fill(logRelative, -20.0);
        PackStarFrequencySeverityPAC.Interval interval =
                PackStarFrequencySeverityPAC.evaluate(
                        logRelative, 1.0, 20.0, 0.025, 0.025);
        PackStarFrequencySeverityPAC.Sizing unreachable =
                PackStarFrequencySeverityPAC.size(
                        new PackStarFrequencySeverityPAC.Moments(
                                0.01, 0.2, 0.5),
                        4000, 400, 0.683, 0.9, 20.0,
                        0.025, 0.025);

        assertFalse(unreachable.reachableAtMax);
        assertFalse(PackStarEstimator.shouldExtendFrequencySeverityDiscovery(
                100, 400, interval, unreachable, 0.683));
    }
}
