/*
** This file is part of OSPREY 3.0
**
** OSPREY Protein Redesign Software Version 3.0
** Copyright (C) 2001-2018 Bruce Donald Lab, Duke University
**
** OSPREY is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License version 2
** as published by the Free Software Foundation.
**
** You should have received a copy of the GNU General Public License
** along with OSPREY.  If not, see <http://www.gnu.org/licenses/>.
*/

package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.confspace.RCTuple;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.energy.EnergyCalculator;
import edu.duke.cs.osprey.energy.ResidueInteractions;
import edu.duke.cs.osprey.energy.forcefield.ResidueForcefieldEnergy;
import edu.duke.cs.osprey.kstar.pfunc.BoltzmannCalculator;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.tools.MathTools;
import edu.duke.cs.osprey.tools.ByteBoundedCache;
import edu.duke.cs.osprey.branchdp.InteractionGraph;
import edu.duke.cs.osprey.branchdp.RootedTreeEdge;
import edu.duke.cs.osprey.branchdp.RootedTreeNode;

import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.nio.charset.StandardCharsets;

/**
 * PACK* ensemble partition-function estimation by learned-proposal
 * importance sampling (classical free-energy perturbation).
 *
 * <p>Manuscript mapping: E_t is the energy returned by the fixed target CCD
 * protocol, E_eta is the fitted proposal energy, and xi = E_t - E_eta.
 * Branch-decomposition tables normalize the proposal and draw independent
 * samples from p_eta. With w = exp(-xi/RT), q_t = q_eta * E_p_eta[w].
 * Learning changes E_eta, not E_t; it aims to improve proposal overlap.</p>
 *
 * <p>Training and on-policy refits precede the frozen final proposal. The
 * pilot selects the final sample size; fresh final samples supply the
 * empirical-Bernstein interval for the clipped component y and the
 * Clopper-Pearson upper bound for the exceedance probability pi.
 * The externally supplied conditional mean-excess bound m_u controls the
 * remaining contribution through m_u * piUpper. The same fixed final batch
 * supplies a rejection diagnostic, not validation of that assumption.</p>
 *
 * <p>Historical configuration keys and diagnostic names
 * are retained for reproducibility; internal names use manuscript terminology.
 * Exact proposal operations refer to the finite factorized model; the
 * implementation uses floating-point arithmetic.</p>
 */
public class PackStarEstimator {

    private static final long MISSING_PAIR_KEY = Long.MIN_VALUE;

    // Configuration
    private static final String PAC_SAMPLES_PROPERTY = "packstar.pac.samples";
    private static final String PAC_CONFIDENCE_PROPERTY = "packstar.pac.confidence";
    private static final String PAC_TARGET_EPSILON_PROPERTY = "packstar.pac.targetEpsilon";
    private static final String PAC_RANDOM_SEED_PROPERTY = "packstar.pac.randomSeed";
    private static final String PAC_SAMPLING_PARALLEL_PROPERTY = "packstar.pac.sampling.parallel";
    private static final String PAC_SAMPLING_THREADS_PROPERTY = "packstar.pac.sampling.threads";
    private static final String PAC_SAMPLING_LARGE_LAMBDA_PROPERTY = "packstar.pac.sampling.largeLambdaThreshold";
    private static final String PAC_SAMPLING_PROGRESS_PROPERTY = "packstar.pac.sampling.progress";
    private static final String PAC_SAMPLING_GPU_PROPERTY = "packstar.pac.sampling.gpu";
    private static final String PAC_CCD_SUBMISSION_BATCH_SIZE_PROPERTY =
            "packstar.pac.ccd.submissionBatchSize";
    private static final String PAC_CCD_CACHE_BYTES_PROPERTY = "packstar.pac.ccd.cacheBytes";
    private static final String PAC_CDF_CACHE_BYTES_PROPERTY = "packstar.pac.sampling.cdfCacheBytes";
    // Train/adaptation/final sizing for the adaptive mean-excess route.
    private static final String PAC_TRAIN_SAMPLES_PROPERTY = "packstar.pac.trainSamples";
    private static final String PAC_PILOT_SAMPLES_PROPERTY = "packstar.pac.pilotSamples";
    private static final String PAC_MAX_EST_SAMPLES_PROPERTY = "packstar.pac.maxEstSamples";
    private static final String PAC_NSTAR_INFLATE_PROPERTY = "packstar.pac.nstarInflate";
    // Adaptive mean-excess: choose a count/context-shrunk eta, repair
    // proposal-support shift with bounded on-policy refits, then freeze the
    // proposal before final samples and the final conditional mean-excess rejection diagnostic.
    private static final String PAC_MEAN_EXCESS_RELATIVE_BOUND_PROPERTY =
            "packstar.pac.frequencySeverity.relativeBoundKcal";
    private static final String PAC_CONDITIONAL_MEAN_EXCESS_BOUND_PROPERTY =
            "packstar.pac.frequencySeverity.severityCap";
    private static final String PAC_MEAN_EXCESS_PREMISE_ID_PROPERTY =
            "packstar.pac.frequencySeverity.severityPremiseId";
    private static final String PAC_MEAN_EXCESS_SHRINK_GRID_PROPERTY =
            "packstar.pac.frequencySeverity.shrinkGrid";
    private static final String PAC_MEAN_EXCESS_MIN_SHIFT_ESS_FRACTION_PROPERTY =
            "packstar.pac.frequencySeverity.minShiftEssFraction";
    private static final String PAC_MEAN_EXCESS_DISCOVERY_MIN_SHIFT_ESS_FRACTION_PROPERTY =
            "packstar.pac.frequencySeverity.discoveryMinShiftEssFraction";
    private static final String PAC_MEAN_EXCESS_MAX_REFITS_PROPERTY =
            "packstar.pac.frequencySeverity.maxRefits";
    private static final String PAC_MEAN_EXCESS_DISCOVERY_SAMPLES_PROPERTY =
            "packstar.pac.frequencySeverity.discoverySamples";
    private static final String PAC_MEAN_EXCESS_DISCOVERY_MAX_SAMPLES_PROPERTY =
            "packstar.pac.frequencySeverity.discoveryMaxSamples";
    private static final String PAC_MEAN_EXCESS_VALIDATION_SAMPLES_PROPERTY =
            "packstar.pac.frequencySeverity.validationSamples";
    private static final String PAC_MEAN_EXCESS_MIN_TRAIN_COUNT_PROPERTY =
            "packstar.pac.frequencySeverity.minTrainCount";
    private static final String PAC_MEAN_EXCESS_MAX_UNDERTRAINED_AMPLIFICATION_PROPERTY =
            "packstar.pac.frequencySeverity.maxUndertrainedAmplification";
    private static final String PAC_MEAN_EXCESS_SIZE_SAFETY_PROPERTY =
            "packstar.pac.frequencySeverity.sizeSafety";
    private static final String PAC_MEAN_EXCESS_TEST_ALPHA_PROPERTY =
            "packstar.pac.frequencySeverity.severityTestAlpha";
    private static final String DP_PARALLEL_THREADS_PROPERTY = "packstar.dp.parallel.threads";
    private static final int DEFAULT_SAMPLES = 1000;
    private static final double DEFAULT_CONFIDENCE = 0.05; // delta = 0.05 => 95% confidence
    private static final double DEFAULT_TARGET_EPSILON = 0.683;
    private static final long DEFAULT_RANDOM_SEED = 42L;
    private static final int DEFAULT_SAMPLING_LARGE_LAMBDA = 65_536;
    private static final double DEFAULT_TRAIN_FRACTION = 0.5;
    private static final double DEFAULT_PILOT_FRACTION = 0.1;
    private static final int DEFAULT_MAX_EST_SAMPLES = 4000;
    private static final double DEFAULT_NSTAR_INFLATE = 1.3;
    private static final int DEFAULT_PAC_CCD_SUBMISSION_BATCH_SIZE = 512;
    private static final double DEFAULT_MEAN_EXCESS_RELATIVE_BOUND_KCAL = 1.0;
    private static final double DEFAULT_CONDITIONAL_MEAN_EXCESS_BOUND = 20.0;
    private static final String DEFAULT_MEAN_EXCESS_PREMISE_ID =
            "conditional-relative-gauge-S0-20-not-externally-recalibrated";
    private static final String DEFAULT_MEAN_EXCESS_SHRINK_GRID =
            "0:0,2:5,5:10,10:20";
    private static final int MEAN_EXCESS_FOLDS = 2;
    // Defaults copied from the successful 42-case mean-excess run;
    // its source-shift diagnostics used permissive 1e-12 ESS thresholds.
    private static final double DEFAULT_MEAN_EXCESS_MIN_SHIFT_ESS_FRACTION = 1.0e-12;
    private static final double DEFAULT_MEAN_EXCESS_DISCOVERY_MIN_SHIFT_ESS_FRACTION = 1.0e-12;
    private static final int DEFAULT_MEAN_EXCESS_MAX_REFITS = 8;
    private static final int DEFAULT_MEAN_EXCESS_DISCOVERY_SAMPLES = 100;
    private static final int DEFAULT_MEAN_EXCESS_DISCOVERY_MAX_SAMPLES = 400;
    private static final int DEFAULT_MEAN_EXCESS_VALIDATION_SAMPLES = 400;
    private static final int DEFAULT_MEAN_EXCESS_MIN_TRAIN_COUNT = 5;
    private static final double DEFAULT_MEAN_EXCESS_MAX_UNDERTRAINED_AMPLIFICATION = 1.25;
    private static final double DEFAULT_MEAN_EXCESS_SIZE_SAFETY = 0.9;
    private static final double DEFAULT_MEAN_EXCESS_TEST_ALPHA = 0.10;
    private static final double PAIR_ETA_RESIDUAL_CAP_KCAL = 3.0;
    // Inputs
    private final RootedTreeNode rootedRoot;
    private final RootedTreeEdge rootedRootEdge;
    private final EnergyMatrix branchMinimizingEmat;
    private final EnergyMatrix branchRigidEmat;
    private final InteractionGraph interactionGraph;
    private final ConfEnergyCalculator minimizingEcalc;
    private final RCs rcs;
    private final double RT;
    /** Sorted immutable base-graph edges used by all hot conformation loops. */
    private final int[][] interactionEdges;
    /** Edge indices grouped by their lower position, preserving old pair-sum order. */
    private final int[][] interactionEdgeIndicesByFirstPosition;
    /** Lazily materialized interaction templates; templates are read-only after creation. */
    private final ResidueInteractions[][] singleInteractionTemplates;
    private final Map<Long, ResidueInteractions> pairInteractionTemplates = new ConcurrentHashMap<>();
    private ResidueInteractions shellInteractionTemplate;
    /**
     * Reusable pair-only corrected matrix.  Pair eta refits mutate only the
     * observed cells; the touched-cell lists let the next refit restore the
     * exact base values without cloning the full EMAT again.
     */
    private EnergyMatrix correctedPairwiseEmat;
    private long[] correctedPairwiseOneBodyKeys = new long[0];
    private int correctedPairwiseOneBodyCount;
    private long[] correctedPairwisePairKeys = new long[0];
    private int correctedPairwisePairCount;
    private double correctedPairwiseGlobalOffsetKcal;
    /** Sampling structure/resources are reused until the proposal tree changes or compute ends. */
    private List<RootedTreeEdge> samplingTopDownOrder;
    private RootedTreeEdge samplingOrderRoot;
    private ExecutorService samplingPool;
    private boolean samplingResourcesClosed;
    /** Reuse conditional-CDF storage on each sampling worker between groups. */
    private final ThreadLocal<double[]> samplingCdfScratch =
            ThreadLocal.withInitial(() -> new double[0]);
    /**
     * Energy matrix represented by the DP tables currently loaded in rootedRoot.
     * CPU ancestral sampling must use this exact matrix for non-leaf local terms;
     * otherwise the conditional weights disagree with the child DP normalizers.
     */
    private EnergyMatrix activeProposalEmat;
    /** These caches belong to one fixed target and one estimator, never another K* state. */
    private final ByteBoundedCache<ConformationKey, CachedTargetEnergy> ccdEnergyCache;
    private final ByteBoundedCache<ConditionalCdfKey, double[]> conditionalCdfCache;
    private final RootedTreeEdge.SamplingRequestCache gpuSamplingRequestCache;

    // Configuration
    private final int numSamples;
    private final double delta; // confidence parameter
    private final double targetEpsilon;
    private final long randomSeed;
    private final String randomStreamIdentity;
    private final boolean gpuSampling;
    private final int samplingThreads;
    private final int samplingLargeLambdaThreshold;
    private final boolean samplingProgress;
    private final int ccdSubmissionBatchSize;
    private final int trainSamples;
    private final int pilotSamples;
    private final int maxEstSamples;
    private final int sampleBudget;
    private final double nstarInflate;
    private final double meanExcessRelativeBoundKcal;
    private final double conditionalMeanExcessBound;
    private final String meanExcessPremiseId;
    private final MeanExcessShrinkPair[] meanExcessShrinkGrid;
    private final double meanExcessMinShiftEssFraction;
    private final double meanExcessDiscoveryMinShiftEssFraction;
    private final int meanExcessMaxRefits;
    private final int meanExcessDiscoverySamples;
    private final int meanExcessDiscoveryMaxSamples;
    private final int meanExcessValidationSamples;
    private final int meanExcessMinTrainCount;
    private final double meanExcessMaxUndertrainedAmplification;
    private final double meanExcessSizeSafety;
    private final double meanExcessTestAlpha;

    // Results
    private BigDecimal zLower;
    private BigDecimal zUpper;
    private double epsilon;
    private double logZLowerPAC;
    private double logZUpperPAC;
    private int totalCCDCalls;
    /** Number of sample records presented to CCD, including duplicates. */
    private int totalCCDSampleRecords;
    private int totalCCDCacheHits;

    // Statistics
    private double cvPsi; // coefficient of variation
    private double meanResidual;
    private double stdResidual;
    private boolean certificateValid = false;
    private String certificateFailureReason = "not computed";

    private static final BoltzmannCalculator LOG_SPACE_EXP =
            new BoltzmannCalculator(PartitionFunction.decimalPrecision);
    private final LocalRCMap[] localRCByGlobalRC;

    public PackStarEstimator(RootedTreeNode rootedRoot,
                                RootedTreeEdge rootedRootEdge,
                                EnergyMatrix branchMinimizingEmat,
                                EnergyMatrix branchRigidEmat,
                                 InteractionGraph interactionGraph,
                                 ConfEnergyCalculator minimizingEcalc,
                                 RCs rcs) {
        this(rootedRoot, rootedRootEdge,
                branchMinimizingEmat, branchRigidEmat,
                interactionGraph, minimizingEcalc,
                rcs,
                Double.NaN);
    }

    public PackStarEstimator(RootedTreeNode rootedRoot,
                                RootedTreeEdge rootedRootEdge,
                                EnergyMatrix branchMinimizingEmat,
                                 EnergyMatrix branchRigidEmat,
                                 InteractionGraph interactionGraph,
                                 ConfEnergyCalculator minimizingEcalc,
                                 RCs rcs,
                                 double requestedTargetEpsilon) {
        this(rootedRoot, rootedRootEdge,
                branchMinimizingEmat, branchRigidEmat,
                interactionGraph, minimizingEcalc,
                rcs,
                requestedTargetEpsilon,
                Integer.MAX_VALUE);
    }

    public PackStarEstimator(RootedTreeNode rootedRoot,
                                RootedTreeEdge rootedRootEdge,
                                EnergyMatrix branchMinimizingEmat,
                                 EnergyMatrix branchRigidEmat,
                                 InteractionGraph interactionGraph,
                                 ConfEnergyCalculator minimizingEcalc,
                                 RCs rcs,
                                 double requestedTargetEpsilon,
                                 int requestedSampleBudget) {
        this(rootedRoot, rootedRootEdge,
                branchMinimizingEmat, branchRigidEmat,
                interactionGraph, minimizingEcalc,
                rcs,
                requestedTargetEpsilon,
                requestedSampleBudget,
                defaultRandomStreamIdentity(rcs));
    }

    /**
     * Construct an estimator with a stable logical random-stream identity.
     *
     * <p>The identity describes the calculation rather than its process-local
     * construction order. K-star supplies a state role plus the exact allowed-RC
     * signature, so shard count and rank-to-node mapping do not change the
     * estimator's random stream.</p>
     */
    public PackStarEstimator(RootedTreeNode rootedRoot,
                                RootedTreeEdge rootedRootEdge,
                                EnergyMatrix branchMinimizingEmat,
                                 EnergyMatrix branchRigidEmat,
                                 InteractionGraph interactionGraph,
                                 ConfEnergyCalculator minimizingEcalc,
                                 RCs rcs,
                                 double requestedTargetEpsilon,
                                 int requestedSampleBudget,
                                 String randomStreamIdentity) {
        this.rootedRoot = rootedRoot;
        this.rootedRootEdge = rootedRootEdge;
        this.branchMinimizingEmat = branchMinimizingEmat;
        this.branchRigidEmat = branchRigidEmat;
        this.interactionGraph = interactionGraph;
        this.minimizingEcalc = minimizingEcalc;
        this.rcs = rcs;
        this.RT = BoltzmannCalculator.RClassic * BoltzmannCalculator.TClassic;
        this.activeProposalEmat = branchMinimizingEmat;
        this.interactionEdges = copySortedInteractionEdges(interactionGraph);
        this.interactionEdgeIndicesByFirstPosition =
                indexInteractionEdgesByFirstPosition(
                        rcs.getNumPos(), interactionEdges);
        this.singleInteractionTemplates = new ResidueInteractions[rcs.getNumPos()][];
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            int maxGlobalRC = -1;
            for (int local = 0; local < rcs.getNum(pos); local++) {
                maxGlobalRC = Math.max(maxGlobalRC, rcs.get(pos, local));
            }
            this.singleInteractionTemplates[pos] =
                    new ResidueInteractions[Math.max(
                            branchMinimizingEmat.getNumConfAtPos(pos),
                            maxGlobalRC + 1)];
        }
        this.sampleBudget = sanitizeSampleBudget(requestedSampleBudget);

        this.numSamples = capSampleCount(
                Math.max(1, getConfigInteger(PAC_SAMPLES_PROPERTY, DEFAULT_SAMPLES)),
                1);
        String deltaStr = getConfigProperty(PAC_CONFIDENCE_PROPERTY, null);
        this.delta = (deltaStr != null) ? Double.parseDouble(deltaStr) : DEFAULT_CONFIDENCE;
        double defaultTarget = Double.isFinite(requestedTargetEpsilon) && requestedTargetEpsilon > 0.0
                ? requestedTargetEpsilon
                : DEFAULT_TARGET_EPSILON;
        this.targetEpsilon = Math.max(0.0,
                getConfigDouble(PAC_TARGET_EPSILON_PROPERTY, defaultTarget));
        long baseRandomSeed = getConfigLong(PAC_RANDOM_SEED_PROPERTY, DEFAULT_RANDOM_SEED);
        this.randomStreamIdentity = requireRandomStreamIdentity(randomStreamIdentity);
        this.randomSeed = deriveRandomSeed(baseRandomSeed, this.randomStreamIdentity);
        this.gpuSampling = getConfigBoolean(PAC_SAMPLING_GPU_PROPERTY, false);
        this.samplingThreads = resolveConfiguredSamplingThreads();
        this.samplingLargeLambdaThreshold = Math.max(1,
                getConfigInteger(PAC_SAMPLING_LARGE_LAMBDA_PROPERTY,
                        DEFAULT_SAMPLING_LARGE_LAMBDA));
        this.samplingProgress = getConfigBoolean(PAC_SAMPLING_PROGRESS_PROPERTY, true);
        this.ccdSubmissionBatchSize = Math.max(1,
                getConfigInteger(PAC_CCD_SUBMISSION_BATCH_SIZE_PROPERTY,
                        DEFAULT_PAC_CCD_SUBMISSION_BATCH_SIZE));
        this.ccdEnergyCache = new ByteBoundedCache<>(Math.max(0L,
                getConfigLong(PAC_CCD_CACHE_BYTES_PROPERTY, 64L << 20)));
        this.conditionalCdfCache = new ByteBoundedCache<>(Math.max(0L,
                getConfigLong(PAC_CDF_CACHE_BYTES_PROPERTY, 64L << 20)));
        this.gpuSamplingRequestCache = new RootedTreeEdge.SamplingRequestCache(Math.max(0L,
                getConfigLong("packstar.pac.sampling.gpu.preparedCacheBytes", 64L << 20)));
        int twoStageBudget = Math.max(6, this.numSamples);
        int defaultTrainSamples = Math.max(2, (int) Math.floor(twoStageBudget * DEFAULT_TRAIN_FRACTION));
        int defaultPilotSamples = Math.max(2, (int) Math.floor(twoStageBudget * DEFAULT_PILOT_FRACTION));
        int defaultMaxEstSamples = Math.max(2, DEFAULT_MAX_EST_SAMPLES);
        this.trainSamples = capSampleCount(
                Math.max(2, getConfigInteger(PAC_TRAIN_SAMPLES_PROPERTY, defaultTrainSamples)),
                2);
        this.pilotSamples = capSampleCount(
                Math.max(2, getConfigInteger(PAC_PILOT_SAMPLES_PROPERTY, defaultPilotSamples)),
                2);
        this.maxEstSamples = capSampleCount(
                Math.max(2, getConfigInteger(PAC_MAX_EST_SAMPLES_PROPERTY, defaultMaxEstSamples)),
                2);
        this.nstarInflate = Math.max(1.0, getConfigDouble(PAC_NSTAR_INFLATE_PROPERTY, DEFAULT_NSTAR_INFLATE));
        this.meanExcessRelativeBoundKcal = getConfigDouble(
                PAC_MEAN_EXCESS_RELATIVE_BOUND_PROPERTY,
                DEFAULT_MEAN_EXCESS_RELATIVE_BOUND_KCAL);
        this.conditionalMeanExcessBound = getConfigDouble(
                PAC_CONDITIONAL_MEAN_EXCESS_BOUND_PROPERTY,
                DEFAULT_CONDITIONAL_MEAN_EXCESS_BOUND);
        this.meanExcessPremiseId = getConfigProperty(
                PAC_MEAN_EXCESS_PREMISE_ID_PROPERTY,
                DEFAULT_MEAN_EXCESS_PREMISE_ID).trim();
        this.meanExcessShrinkGrid = parseFrequencySeverityShrinkGrid(getConfigProperty(
                PAC_MEAN_EXCESS_SHRINK_GRID_PROPERTY,
                DEFAULT_MEAN_EXCESS_SHRINK_GRID));
        this.meanExcessMinShiftEssFraction = getConfigDouble(
                PAC_MEAN_EXCESS_MIN_SHIFT_ESS_FRACTION_PROPERTY,
                DEFAULT_MEAN_EXCESS_MIN_SHIFT_ESS_FRACTION);
        this.meanExcessDiscoveryMinShiftEssFraction = getConfigDouble(
                PAC_MEAN_EXCESS_DISCOVERY_MIN_SHIFT_ESS_FRACTION_PROPERTY,
                DEFAULT_MEAN_EXCESS_DISCOVERY_MIN_SHIFT_ESS_FRACTION);
        this.meanExcessMaxRefits = getConfigInteger(
                PAC_MEAN_EXCESS_MAX_REFITS_PROPERTY,
                DEFAULT_MEAN_EXCESS_MAX_REFITS);
        this.meanExcessDiscoverySamples = capSampleCount(Math.max(2,
                getConfigInteger(PAC_MEAN_EXCESS_DISCOVERY_SAMPLES_PROPERTY,
                        DEFAULT_MEAN_EXCESS_DISCOVERY_SAMPLES)), 2);
        this.meanExcessDiscoveryMaxSamples = capSampleCount(Math.max(2,
                getConfigInteger(PAC_MEAN_EXCESS_DISCOVERY_MAX_SAMPLES_PROPERTY,
                        DEFAULT_MEAN_EXCESS_DISCOVERY_MAX_SAMPLES)), 2);
        this.meanExcessValidationSamples = capSampleCount(Math.max(2,
                getConfigInteger(PAC_MEAN_EXCESS_VALIDATION_SAMPLES_PROPERTY,
                        DEFAULT_MEAN_EXCESS_VALIDATION_SAMPLES)), 2);
        this.meanExcessMinTrainCount = getConfigInteger(
                PAC_MEAN_EXCESS_MIN_TRAIN_COUNT_PROPERTY,
                DEFAULT_MEAN_EXCESS_MIN_TRAIN_COUNT);
        this.meanExcessMaxUndertrainedAmplification = getConfigDouble(
                PAC_MEAN_EXCESS_MAX_UNDERTRAINED_AMPLIFICATION_PROPERTY,
                DEFAULT_MEAN_EXCESS_MAX_UNDERTRAINED_AMPLIFICATION);
        this.meanExcessSizeSafety = getConfigDouble(
                PAC_MEAN_EXCESS_SIZE_SAFETY_PROPERTY,
                DEFAULT_MEAN_EXCESS_SIZE_SAFETY);
        this.meanExcessTestAlpha = getConfigDouble(
                PAC_MEAN_EXCESS_TEST_ALPHA_PROPERTY,
                DEFAULT_MEAN_EXCESS_TEST_ALPHA);
        if (!Double.isFinite(meanExcessRelativeBoundKcal)
                || !(meanExcessRelativeBoundKcal > 0.0)) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess relative bound must be finite and positive: "
                            + meanExcessRelativeBoundKcal);
        }
        if (!Double.isFinite(conditionalMeanExcessBound)
                || conditionalMeanExcessBound < 0.0) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess conditional mean-excess bound must be finite and nonnegative: "
                            + conditionalMeanExcessBound);
        }
        if (meanExcessPremiseId.isEmpty()) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess conditional mean-excess assumption id must not be empty");
        }
        if (!Double.isFinite(meanExcessMinShiftEssFraction)
                || !(meanExcessMinShiftEssFraction > 0.0)
                || meanExcessMinShiftEssFraction > 1.0) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess minimum shift ESS fraction must be in (0,1]: "
                            + meanExcessMinShiftEssFraction);
        }
        if (!Double.isFinite(meanExcessDiscoveryMinShiftEssFraction)
                || !(meanExcessDiscoveryMinShiftEssFraction > 0.0)
                || meanExcessDiscoveryMinShiftEssFraction
                > meanExcessMinShiftEssFraction) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess discovery shift ESS fraction must be in"
                            + " (0, minimum final shift ESS fraction]: "
                            + meanExcessDiscoveryMinShiftEssFraction);
        }
        if (meanExcessMaxRefits < 0
                || meanExcessMaxRefits > 16) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess maximum refits must be in [0,16]: "
                            + meanExcessMaxRefits);
        }
        if (meanExcessDiscoveryMaxSamples
                < meanExcessDiscoverySamples) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess maximum discovery samples must"
                            + " be at least the initial discovery samples: "
                            + meanExcessDiscoveryMaxSamples + " < "
                            + meanExcessDiscoverySamples);
        }
        if (meanExcessMinTrainCount < 1) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess minimum train count must be positive: "
                            + meanExcessMinTrainCount);
        }
        if (!Double.isFinite(meanExcessMaxUndertrainedAmplification)
                || meanExcessMaxUndertrainedAmplification < 1.0) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess maximum undertrained amplification must be finite and >=1: "
                            + meanExcessMaxUndertrainedAmplification);
        }
        if (!Double.isFinite(meanExcessSizeSafety)
                || !(meanExcessSizeSafety > 0.0)
                || meanExcessSizeSafety > 1.0) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess sizing safety must be in (0,1]: "
                            + meanExcessSizeSafety);
        }
        if (!Double.isFinite(meanExcessTestAlpha)
                || !(meanExcessTestAlpha > 0.0)
                || meanExcessTestAlpha >= 1.0) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess mean-excess diagnostic alpha must be in (0,1): "
                            + meanExcessTestAlpha);
        }
        if (trainSamples < 4) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess two-fold selection requires at least four training samples");
        }
        this.localRCByGlobalRC = buildLocalRCMaps(rcs);
    }

    private int sanitizeSampleBudget(int requestedSampleBudget) {
        return requestedSampleBudget > 0 ? requestedSampleBudget : Integer.MAX_VALUE;
    }

    private int capSampleCount(int configured, int minimum) {
        int capped = sampleBudget == Integer.MAX_VALUE
                ? configured
                : Math.min(configured, Math.max(minimum, sampleBudget));
        return Math.max(minimum, capped);
    }

    private static class CorrectedDPResult {
        final double logZCorrected;

        CorrectedDPResult(double logZCorrected) {
            this.logZCorrected = logZCorrected;
        }
    }

    static final class MeanExcessShrinkPair {
        final double unary;
        final double pair;

        MeanExcessShrinkPair(double unary, double pair) {
            this.unary = unary;
            this.pair = pair;
        }

        String id() {
            return String.format(Locale.ROOT, "ku-%g-kp-%g", unary, pair);
        }
    }

    private static class LocalRCMap {
        final int[] dense;
        final Map<Integer, Integer> sparse;

        LocalRCMap(int[] dense, Map<Integer, Integer> sparse) {
            this.dense = dense;
            this.sparse = sparse;
        }

        int get(int globalRC) {
            if (dense != null) {
                if (globalRC >= 0 && globalRC < dense.length) {
                    return dense[globalRC];
                }
                return -1;
            }
            Integer local = sparse.get(globalRC);
            return local == null ? -1 : local;
        }
    }

    private static class SampleGroup {
        final long mIdx;
        /** Local M digits captured while grouping; avoids a second decode. */
        int[] mRCs;
        int[] sampleIndices = new int[4];
        int count = 0;

        SampleGroup(long mIdx) {
            this.mIdx = mIdx;
        }

        void add(int sampleIndex) {
            if (count >= sampleIndices.length) {
                sampleIndices = Arrays.copyOf(sampleIndices, sampleIndices.length * 2);
            }
            sampleIndices[count++] = sampleIndex;
        }
    }

    private static int getConfigInteger(String key, int defaultValue) {
        return PackStarConfig.getInteger(key, defaultValue, "[PACK*]");
    }

    private static long getConfigLong(String key, long defaultValue) {
        return PackStarConfig.getLong(key, defaultValue, "[PACK*]");
    }

    private static double getConfigDouble(String key, double defaultValue) {
        return PackStarConfig.getDouble(key, defaultValue, "[PACK*]");
    }

    private static boolean getConfigBoolean(String key, boolean defaultValue) {
        return PackStarConfig.getBoolean(key, defaultValue);
    }

    private static String getConfigProperty(String key, String defaultValue) {
        return PackStarConfig.getProperty(key, defaultValue);
    }

    private static double clamp(double value, double lower, double upper) {
        if (Double.isNaN(value)) return lower;
        return Math.max(lower, Math.min(upper, value));
    }

    static MeanExcessShrinkPair[] parseFrequencySeverityShrinkGrid(String configured) {
        if (configured == null || configured.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess shrink grid must not be empty");
        }
        List<MeanExcessShrinkPair> parsed = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        boolean containsRaw = false;
        for (String token : configured.split(",")) {
            String trimmed = token.trim();
            String[] parts = trimmed.split(":", -1);
            if (parts.length != 2) {
                throw new IllegalArgumentException(
                        "PACK* mean-excess shrink values must have unary:pair form: "
                                + trimmed);
            }
            final double unary;
            final double pair;
            try {
                unary = Double.parseDouble(parts[0].trim());
                pair = Double.parseDouble(parts[1].trim());
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException(
                        "Invalid PACK* mean-excess shrink value: " + trimmed, ex);
            }
            if (!Double.isFinite(unary) || unary < 0.0
                    || !Double.isFinite(pair) || pair < 0.0) {
                throw new IllegalArgumentException(
                        "PACK* mean-excess shrink strengths must be finite and nonnegative: "
                                + trimmed);
            }
            String key = Long.toHexString(Double.doubleToLongBits(unary))
                    + ":" + Long.toHexString(Double.doubleToLongBits(pair));
            if (!seen.add(key)) continue;
            parsed.add(new MeanExcessShrinkPair(unary, pair));
            if (unary == 0.0 && pair == 0.0) containsRaw = true;
        }
        if (parsed.isEmpty() || !containsRaw) {
            throw new IllegalArgumentException(
                    "PACK* mean-excess shrink grid must contain raw eta as 0:0");
        }
        return parsed.toArray(new MeanExcessShrinkPair[0]);
    }

    private static int resolveConfiguredSamplingThreads() {
        if (!getConfigBoolean(PAC_SAMPLING_PARALLEL_PROPERTY, true)) {
            return 1;
        }

        int configured = getConfigInteger(PAC_SAMPLING_THREADS_PROPERTY, 0);
        if (configured <= 0) {
            configured = getConfigInteger(DP_PARALLEL_THREADS_PROPERTY, 0);
        }
        if (configured <= 0) {
            configured = Runtime.getRuntime().availableProcessors();
        }
        return Math.max(1, configured);
    }

    private static LocalRCMap[] buildLocalRCMaps(RCs rcs) {
        LocalRCMap[] maps = new LocalRCMap[rcs.getNumPos()];
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            int numRCs = rcs.getNum(pos);
            int maxRC = -1;
            for (int local = 0; local < numRCs; local++) {
                maxRC = Math.max(maxRC, rcs.get(pos, local));
            }

            if (maxRC >= 0 && maxRC <= Math.max(4096, numRCs * 8)) {
                int[] dense = new int[maxRC + 1];
                Arrays.fill(dense, -1);
                for (int local = 0; local < numRCs; local++) {
                    dense[rcs.get(pos, local)] = local;
                }
                maps[pos] = new LocalRCMap(dense, null);
            } else {
                Map<Integer, Integer> sparse = new HashMap<>(Math.max(16, numRCs * 2));
                for (int local = 0; local < numRCs; local++) {
                    sparse.put(rcs.get(pos, local), local);
                }
                maps[pos] = new LocalRCMap(null, sparse);
            }
        }
        return maps;
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }

    static long deriveRandomSeed(long baseRandomSeed, String randomStreamIdentity) {
        String identity = requireRandomStreamIdentity(randomStreamIdentity);
        long identityHash = 0xcbf29ce484222325L;
        for (byte value : identity.getBytes(StandardCharsets.UTF_8)) {
            identityHash ^= value & 0xffL;
            identityHash *= 0x100000001b3L;
        }
        return mix64(baseRandomSeed ^ identityHash);
    }

    private Random stageRandom(String stage) {
        return new Random(deriveRandomSeed(
                randomSeed, randomStreamIdentity + "|unconditional-v1|" + stage));
    }

    private static String requireRandomStreamIdentity(String identity) {
        if (identity == null || identity.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "PACK* random-stream identity is required");
        }
        return identity;
    }

    private static String defaultRandomStreamIdentity(RCs rcs) {
        StringBuilder identity = new StringBuilder(
                "packstar|standalone|pac-v1");
        identity.append('|').append(rcs.getNumPos());
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            identity.append('|').append(pos).append(':');
            for (int rc : rcs.get(pos)) {
                identity.append(rc).append(',');
            }
        }
        return identity.toString();
    }

    private static ThreadFactory daemonThreadFactory(String namePrefix) {
        AtomicInteger counter = new AtomicInteger(0);
        return runnable -> {
            Thread thread = new Thread(runnable, namePrefix + "-" + counter.getAndIncrement());
            thread.setDaemon(true);
            return thread;
        };
    }

    private static int[][] copySortedInteractionEdges(InteractionGraph graph) {
        List<int[]> edges = new ArrayList<>(graph.getEdgeList().size());
        for (int[] edge : graph.getEdgeList()) {
            if (edge == null || edge.length < 2) {
                throw new IllegalArgumentException(
                        "interaction graph contains a malformed edge");
            }
            int pos1 = Math.min(edge[0], edge[1]);
            int pos2 = Math.max(edge[0], edge[1]);
            edges.add(new int[]{pos1, pos2});
        }
        edges.sort((a, b) -> {
            int compare = Integer.compare(a[0], b[0]);
            return compare != 0 ? compare : Integer.compare(a[1], b[1]);
        });
        // InteractionGraph.hasEdge() is set-like.  Keep the same semantics if
        // a caller supplied a duplicate edge, otherwise the cached edge list
        // would silently double-count every pair term.
        List<int[]> unique = new ArrayList<>(edges.size());
        int previousPos1 = -1;
        int previousPos2 = -1;
        for (int[] edge : edges) {
            if (edge[0] == previousPos1 && edge[1] == previousPos2) {
                continue;
            }
            unique.add(edge);
            previousPos1 = edge[0];
            previousPos2 = edge[1];
        }
        return unique.toArray(new int[0][]);
    }

    private static int[][] indexInteractionEdgesByFirstPosition(
            int numPositions, int[][] edges) {
        List<Integer>[] byPosition = new List[numPositions];
        for (int pos = 0; pos < numPositions; pos++) {
            byPosition[pos] = new ArrayList<>();
        }
        for (int edgeIndex = 0; edgeIndex < edges.length; edgeIndex++) {
            byPosition[edges[edgeIndex][0]].add(edgeIndex);
        }
        int[][] result = new int[numPositions][];
        for (int pos = 0; pos < numPositions; pos++) {
            result[pos] = byPosition[pos].stream()
                    .mapToInt(Integer::intValue).toArray();
        }
        return result;
    }

    private ResidueInteractions getSingleInteractionTemplate(int pos, int rc) {
        if (pos < 0 || pos >= singleInteractionTemplates.length
                || rc < 0 || rc >= singleInteractionTemplates[pos].length) {
            throw new IllegalArgumentException(
                    "single interaction template index out of range: pos="
                            + pos + " rc=" + rc);
        }
        ResidueInteractions template = singleInteractionTemplates[pos][rc];
        if (template == null) {
            template = minimizingEcalc.makeSingleInters(pos, rc);
            singleInteractionTemplates[pos][rc] = template;
        }
        return template;
    }

    private static long packInteractionTemplateKey(
            int pos1, int rc1, int pos2, int rc2) {
        if (pos1 > pos2) {
            int tmp = pos1;
            pos1 = pos2;
            pos2 = tmp;
            tmp = rc1;
            rc1 = rc2;
            rc2 = tmp;
        }
        return ((long) pos1 << 48) | ((long) rc1 << 32)
                | ((long) pos2 << 16) | (rc2 & 0xffffL);
    }

    private ResidueInteractions getPairInteractionTemplate(
            int pos1, int rc1, int pos2, int rc2) {
        long key = packInteractionTemplateKey(pos1, rc1, pos2, rc2);
        ResidueInteractions template = pairInteractionTemplates.get(key);
        if (template == null) {
            template = minimizingEcalc.makePairInters(pos1, rc1, pos2, rc2);
            pairInteractionTemplates.put(key, template);
        }
        return template;
    }

    private ResidueInteractions getShellInteractionTemplate() {
        if (shellInteractionTemplate == null) {
            shellInteractionTemplate = minimizingEcalc.makeShellInters();
        }
        return shellInteractionTemplate;
    }

    /**
     * Run the full PAC estimation pipeline.
     * Returns the epsilon achieved.
     */
    public double compute() {
        long startTime = System.currentTimeMillis();

        // A PackStarEstimator is normally used once, but resetting these
        // counters makes repeated compute() calls well-defined and keeps the
        // reported CCD savings tied to this run only.
        totalCCDCalls = 0;
        totalCCDSampleRecords = 0;
        totalCCDCacheHits = 0;
        ccdEnergyCache.clear();
        invalidateSamplingCaches();

        System.out.println("[PACK*] random stream: seed="
                + Long.toUnsignedString(randomSeed)
                + ", identityHash="
                + Long.toUnsignedString(
                deriveRandomSeed(0L, randomStreamIdentity), 16));

        // Phase 0: Z_min from existing DP (already computed)
        double logZMin = rootedRootEdge.getLogZUpper(0);
        double logZRigid = rootedRootEdge.getLogZLower(0);
        System.out.println("[PACK*] Phase 0: logZ_min=" + String.format("%.4f", logZMin)
                + ", logZ_rigid=" + String.format("%.4f", logZRigid)
                + ", gap=" + String.format("%.4f", logZMin - logZRigid));

        if (logZMin == Double.NEGATIVE_INFINITY) {

            setZeroBounds("initial DP upper bound is zero");
            System.out.println("[PACK*] Degenerate zero-mass state; skipping sampling and returning epsilon=1.0");
            return epsilon;
        }
        if (!Double.isFinite(logZMin)) {

            setZeroBounds("initial DP upper bound is non-finite: " + logZMin);
            System.out.println("[PACK*] Degenerate non-finite state; skipping sampling and returning epsilon=1.0");
            return epsilon;
        }

        System.out.println("[PACK*-adaptive-frequency-severity] fixed production route"
                + ", samples(train/pilot/maxEst)="
                + trainSamples + "/" + pilotSamples + "/" + maxEstSamples
                + ", severityTestStage=final"
                + ", sampleBudget="
                + (sampleBudget == Integer.MAX_VALUE ? "unbounded" : sampleBudget)
                + ", B_rel=" + meanExcessRelativeBoundKcal
                + ", S0=" + conditionalMeanExcessBound
                + ", maxRefits=" + meanExcessMaxRefits);
        try {
            runAdaptiveFrequencySeverityPAC(startTime);
        } finally {
            shutdownSamplingResources();
        }

        return epsilon;
    }

    private void printFinalSummary(long startTime) {
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("[PACK*] Total: " + totalTime + " ms, " + totalCCDCalls
                + " unique CCD calls for " + totalCCDSampleRecords
                + " sample records, epsilon=" + String.format("%.6f", epsilon)
                + ", confidence=" + String.format("%.2f%%", (1.0 - delta) * 100));
        System.out.println("[PACK*] Z bounds: lower="
                + String.format(Locale.ROOT, "%.6e", zLower)
                + ", upper="
                + String.format(Locale.ROOT, "%.6e", zUpper)
                + ", log10Lower=" + formatLog10(logZLowerPAC)
                + ", log10Upper=" + formatLog10(logZUpperPAC));
    }

    // ========== Adaptive mean-excess estimator ==========

    /**
     * Adaptive mean-excess PAC estimator.
     *
     *   Z = q_eta * E_{p_eta}[ exp(-xi/RT) ],   xi = E_true - E_eta   (exact identity)
     *
     * The q_m training batch is cross-fitted to choose a count/context-shrunk
     * eta.  Bounded on-policy discovery refits repair proposal-support shift;
     * the selected proposal is then frozen before validation and final
     * samples. The pilot sizes the fresh final sample once the proposal
     * is frozen; there is no alternate conditional/unconditional estimator and
     * no proposal restore after the final selection.
     *
     * Sample-splitting: eta and p_eta depend only on training/adaptation data,
     * while final samples are fresh. Conditional on the frozen
     * conditional mean-excess assumption and the pilot, the final sample size is fixed for the
     * empirical-Bernstein PAC calculation. The same fixed final batch also
     * supplies one conditional mean-excess rejection diagnostic; rejection suppresses the estimate.
     */
    private static class MeanExcessEtaCoverage {
        final int[][] unaryDistinctContexts;
        final EtaCorrections.PairCountMap pairDistinctContexts;

        MeanExcessEtaCoverage(int numPos, int[] numRCs) {
            unaryDistinctContexts = new int[numPos][];
            for (int pos = 0; pos < numPos; pos++) {
                unaryDistinctContexts[pos] = new int[numRCs[pos]];
            }
            pairDistinctContexts = new EtaCorrections.PairCountMap();
        }

        int unary(int pos, int rc) {
            return unaryDistinctContexts[pos][rc];
        }

        int pair(int pos1, int rc1, int pos2, int rc2) {
            return pairDistinctContexts.getOrDefault(
                    EtaCorrections.packPairKey(pos1, rc1, pos2, rc2), 0);
        }
    }

    private static class MeanExcessEtaTraining {
        final EtaCorrections all;
        final EtaCorrections[] folds;
        final MeanExcessEtaCoverage allCoverage;
        final MeanExcessEtaCoverage[] foldCoverage;
        final List<CCDResult> allFitSamples;
        final List<CCDResult>[] foldFitSamples;

        MeanExcessEtaTraining(EtaCorrections all, EtaCorrections[] folds,
                      MeanExcessEtaCoverage allCoverage,
                      MeanExcessEtaCoverage[] foldCoverage,
                      List<CCDResult> allFitSamples,
                      List<CCDResult>[] foldFitSamples) {
            this.all = all;
            this.folds = folds;
            this.allCoverage = allCoverage;
            this.foldCoverage = foldCoverage;
            this.allFitSamples = allFitSamples;
            this.foldFitSamples = foldFitSamples;
        }
    }

    private static class MeanExcessFoldScore {
        final int sampleCount;
        final double logMu;
        final double shiftEss;
        final double shiftEssFraction;
        final double targetM2;
        final double clippedMean;
        final double clippedVariance;
        final double exceedanceProbability;
        final double baselineUnaryUndertrainedMass;
        final double candidateUnaryUndertrainedMass;
        final double baselinePairUndertrainedMass;
        final double candidatePairUndertrainedMass;
        final boolean finite;
        final boolean shiftPass;

        MeanExcessFoldScore(int sampleCount,
                    double logMu, double shiftEss,
                    double shiftEssFraction, double targetM2,
                    double clippedMean,
                    double clippedVariance, double exceedanceProbability,
                    double baselineUnaryUndertrainedMass,
                    double candidateUnaryUndertrainedMass,
                    double baselinePairUndertrainedMass,
                    double candidatePairUndertrainedMass,
                    boolean finite, boolean shiftPass) {
            this.sampleCount = sampleCount;
            this.logMu = logMu;
            this.shiftEss = shiftEss;
            this.shiftEssFraction = shiftEssFraction;
            this.targetM2 = targetM2;
            this.clippedMean = clippedMean;
            this.clippedVariance = clippedVariance;
            this.exceedanceProbability = exceedanceProbability;
            this.baselineUnaryUndertrainedMass =
                    baselineUnaryUndertrainedMass;
            this.candidateUnaryUndertrainedMass =
                    candidateUnaryUndertrainedMass;
            this.baselinePairUndertrainedMass =
                    baselinePairUndertrainedMass;
            this.candidatePairUndertrainedMass =
                    candidatePairUndertrainedMass;
            this.finite = finite;
            this.shiftPass = shiftPass;
        }

    }

    /**
     * Sample-size-weighted out-of-fold model-selection score.  This pools the
     * two independently validated eta fits for a less noisy adaptation
     * diagnostic; it is not a confidence interval and is always followed by a
     * full-refit audit and an independent on-policy pilot.
     */
    private static class MeanExcessCrossfitScore {
        final double shiftEssFraction;
        final double targetM2;
        final double unaryUndertrainedAmplification;
        final double pairUndertrainedAmplification;
        final PackStarFrequencySeverityPAC.Sizing sizing;
        final boolean allFoldsFinite;
        final boolean allFoldsShiftPass;
        final boolean finite;
        final boolean shiftPass;
        final boolean bulkPass;
        final boolean reachabilityPass;

        MeanExcessCrossfitScore(
                double shiftEssFraction,
                double targetM2, double unaryUndertrainedAmplification,
                double pairUndertrainedAmplification,
                PackStarFrequencySeverityPAC.Sizing sizing,
                boolean allFoldsFinite,
                boolean allFoldsShiftPass,
                boolean finite, boolean shiftPass,
                boolean bulkPass,
                boolean reachabilityPass) {
            this.shiftEssFraction = shiftEssFraction;
            this.targetM2 = targetM2;
            this.unaryUndertrainedAmplification =
                    unaryUndertrainedAmplification;
            this.pairUndertrainedAmplification =
                    pairUndertrainedAmplification;
            this.sizing = sizing;
            this.allFoldsFinite = allFoldsFinite;
            this.allFoldsShiftPass = allFoldsShiftPass;
            this.finite = finite;
            this.shiftPass = shiftPass;
            this.bulkPass = bulkPass;
            this.reachabilityPass = reachabilityPass;
        }

        boolean eligible() {
            // Half-fold coverage is deliberately diagnostic: fitting on 250
            // observations must not be required to have the same sparse-cell
            // support as the actual all-500 refit.  The all-training coverage
            // audit remains a hard gate below.
            return allFoldsFinite && allFoldsShiftPass
                    && finite && shiftPass && bulkPass
                    && reachabilityPass;
        }
    }

    private static class MeanExcessCandidateScore {
        final String id;
        final MeanExcessShrinkPair shrink;
        final EtaCorrections fullEta;
        final MeanExcessFoldScore[] folds;
        final MeanExcessCrossfitScore crossfit;
        final boolean crossfitEligible;
        final MeanExcessRefitAudit refitAudit;
        final boolean eligible;

        MeanExcessCandidateScore(String id, MeanExcessShrinkPair shrink,
                         EtaCorrections fullEta,
                         MeanExcessFoldScore[] folds,
                         MeanExcessCrossfitScore crossfit,
                         MeanExcessRefitAudit refitAudit) {
            this.id = id;
            this.shrink = shrink;
            this.fullEta = fullEta;
            this.folds = folds;
            this.crossfit = crossfit;
            this.refitAudit = refitAudit;
            this.crossfitEligible = crossfit != null && crossfit.eligible();
            this.eligible = crossfitEligible && refitAudit != null
                    && refitAudit.eligible();
        }
    }

    private static class MeanExcessRefitAudit {
        final double shiftEssFraction;
        final boolean finite;
        final boolean shiftPass;
        final boolean coveragePass;

        MeanExcessRefitAudit(
                double shiftEssFraction,
                boolean finite, boolean shiftPass,
                boolean coveragePass) {
            this.shiftEssFraction = shiftEssFraction;
            this.finite = finite;
            this.shiftPass = shiftPass;
            this.coveragePass = coveragePass;
        }

        boolean eligible() {
            return finite && shiftPass && coveragePass;
        }
    }

    private static class MeanExcessPilotEvaluation {
        final PackStarFrequencySeverityPAC.Interval interval;
        final PackStarFrequencySeverityPAC.Sizing sizing;

        MeanExcessPilotEvaluation(
                PackStarFrequencySeverityPAC.Interval interval,
                PackStarFrequencySeverityPAC.Sizing sizing) {
            this.interval = interval;
            this.sizing = sizing;
        }
    }

    /** Scratch arrays reused across sequential candidate/fold scoring. */
    private static class MeanExcessFoldScratch {
        double[] logA = new double[0];
        double[] logV = new double[0];
        double[] proposalWeights = new double[0];
        double[] logRelative = new double[0];
        double[] bulk = new double[0];

        void ensureCapacity(int n) {
            if (logA.length >= n) return;
            logA = new double[n];
            logV = new double[n];
            proposalWeights = new double[n];
            logRelative = new double[n];
            bulk = new double[n];
        }
    }

    /** Cache fitted eta energies for one candidate-scoring pass. */
    private class MeanExcessEtaEnergyCache {
        private static class CachedEnergyVector {
            final double[] values;
            final boolean[] computed;

            CachedEnergyVector(int size) {
                values = new double[size];
                computed = new boolean[size];
            }
        }

        private final IdentityHashMap<CCDResult, Integer> sampleIndices =
                new IdentityHashMap<>();
        private final IdentityHashMap<EtaCorrections, CachedEnergyVector> energies =
                new IdentityHashMap<>();

        MeanExcessEtaEnergyCache(List<CCDResult> trainingSamples,
                List<CCDResult>[] validationFolds, MeanExcessEtaTraining training) {
            addSamples(trainingSamples);
            addSamples(training.allFitSamples);
            for (List<CCDResult> samples : training.foldFitSamples) addSamples(samples);
            for (List<CCDResult> samples : validationFolds) addSamples(samples);
        }

        private void addSamples(List<CCDResult> samples) {
            if (samples == null) return;
            for (CCDResult sample : samples) {
                if (!sampleIndices.containsKey(sample))
                    sampleIndices.put(sample, sampleIndices.size());
            }
        }

        double getEnergy(CCDResult sample, EtaCorrections eta) {
            Integer index = sampleIndices.get(sample);
            if (index == null)
                throw new IllegalArgumentException("eta energy cache does not contain CCD sample");
            CachedEnergyVector vector = energies.get(eta);
            if (vector == null) {
                vector = new CachedEnergyVector(sampleIndices.size());
                energies.put(eta, vector);
            }
            if (!vector.computed[index]) {
                vector.values[index] = computeEtaEnergy(sample, eta);
                // Preserve non-finite values for every subsequent validation.
                vector.computed[index] = true;
            }
            return vector.values[index];
        }
    }

    private static class MeanExcessCoverageFlags {
        final boolean[] unaryUndertrained;
        final boolean[] pairUndertrained;

        MeanExcessCoverageFlags(int sampleCount) {
            unaryUndertrained = new boolean[sampleCount];
            pairUndertrained = new boolean[sampleCount];
        }
    }

    private MeanExcessCoverageFlags buildFrequencySeverityCoverageFlags(
            List<CCDResult> samples, MeanExcessEtaCoverage coverage) {
        MeanExcessCoverageFlags flags =
                new MeanExcessCoverageFlags(samples.size());
        for (int i = 0; i < samples.size(); i++) {
            int[] conf = samples.get(i).conf;
            flags.unaryUndertrained[i] =
                    touchesFrequencySeverityUndertrainedUnary(conf, coverage);
            flags.pairUndertrained[i] =
                    touchesFrequencySeverityUndertrainedPair(conf, coverage);
        }
        return flags;
    }

    private List<MeanExcessCandidateScore> scoreFrequencySeverityEtaCandidates(
            List<CCDResult> trainingSamples,
            MeanExcessEtaTraining training) {
        List<CCDResult>[] validationFolds = new List[MEAN_EXCESS_FOLDS];
        MeanExcessCoverageFlags[] validationCoverageFlags =
                new MeanExcessCoverageFlags[MEAN_EXCESS_FOLDS];
        for (int fold = 0; fold < MEAN_EXCESS_FOLDS; fold++) {
            validationFolds[fold] = new ArrayList<>();
        }
        for (int index = 0; index < trainingSamples.size(); index++) {
            validationFolds[index % MEAN_EXCESS_FOLDS].add(trainingSamples.get(index));
        }
        for (int validationFold = 0;
             validationFold < MEAN_EXCESS_FOLDS; validationFold++) {
            int fitFold = 1 - validationFold;
            validationCoverageFlags[validationFold] =
                    buildFrequencySeverityCoverageFlags(
                            validationFolds[validationFold],
                            training.foldCoverage[fitFold]);
        }
        MeanExcessCoverageFlags allCoverageFlags =
                buildFrequencySeverityCoverageFlags(
                        trainingSamples, training.allCoverage);

        List<MeanExcessCandidateScore> candidates = new ArrayList<>();
        MeanExcessFoldScratch foldScratch =
                new MeanExcessFoldScratch();
        MeanExcessEtaEnergyCache energyCache =
                new MeanExcessEtaEnergyCache(
                        trainingSamples, validationFolds, training);
        // Crossfit the graph-preserving pair correction at fixed alpha one.
        for (MeanExcessShrinkPair shrink : meanExcessShrinkGrid) {
            EtaCorrections fullPairEta = shrinkEta(
                    training.all, training.allCoverage,
                    shrink.unary, shrink.pair);
            fullPairEta = refitFrequencySeverityPairEta(training.allFitSamples,
                    fullPairEta, training.allCoverage, shrink);
            EtaCorrections[] foldPairEta =
                    new EtaCorrections[MEAN_EXCESS_FOLDS];
            for (int fitFold = 0;
                 fitFold < MEAN_EXCESS_FOLDS; fitFold++) {
                foldPairEta[fitFold] = shrinkEta(
                        training.folds[fitFold],
                        training.foldCoverage[fitFold],
                        shrink.unary, shrink.pair);
                foldPairEta[fitFold] = refitFrequencySeverityPairEta(
                        training.foldFitSamples[fitFold], foldPairEta[fitFold],
                        training.foldCoverage[fitFold], shrink);
            }

            MeanExcessFoldScore[] pairFoldScores =
                    new MeanExcessFoldScore[MEAN_EXCESS_FOLDS];
            for (int validationFold = 0;
                 validationFold < MEAN_EXCESS_FOLDS; validationFold++) {
                int fitFold = 1 - validationFold;
                pairFoldScores[validationFold] =
                        scoreFrequencySeverityEtaFold(
                        validationFolds[validationFold],
                        foldPairEta[fitFold],
                        foldScratch,
                        validationCoverageFlags[validationFold],
                        energyCache);
            }
            MeanExcessCrossfitScore pairCrossfit =
                    poolFrequencySeverityFoldScores(pairFoldScores);
            MeanExcessRefitAudit pairRefitAudit =
                    auditFrequencySeverityRefit(
                            trainingSamples, fullPairEta,
                            allCoverageFlags, foldScratch, energyCache);
            candidates.add(new MeanExcessCandidateScore(
                    meanExcessCandidateId(shrink),
                    shrink, fullPairEta,
                    pairFoldScores, pairCrossfit, pairRefitAudit));
        }
        return candidates;
    }

    static String meanExcessCandidateId(MeanExcessShrinkPair shrink) {
        return shrink.id() + "-alpha-1.00000-pair-only";
    }

    private PackStarProposalLearning.Data proposalLearningData(
            List<CCDResult> samples, EtaCorrections pair) {
        int[][] conf = new int[samples.size()][];
        double[] residual = new double[samples.size()], logWeight = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            CCDResult sample = samples.get(i);
            double energy = sample.eMin + computeEtaEnergy(sample, pair);
            conf[i] = sample.conf;
            residual[i] = sample.eTrue - energy;
            logWeight[i] = meanExcessSourceLogWeight(sample, energy);
        }
        return new PackStarProposalLearning.Data(conf, residual, logWeight);
    }

    /** Jointly fit full CCD residual using only cells of the existing graph. */
    private EtaCorrections refitFrequencySeverityPairEta(List<CCDResult> samples,
            EtaCorrections prior, MeanExcessEtaCoverage coverage,
            MeanExcessShrinkPair shrink) {
        if (samples.size() < 2) return prior;
        int[][] unaryIds = new int[prior.oneBody.length][];
        List<Double> ratios = new ArrayList<>();
        for (int p = 0; p < unaryIds.length; p++) {
            unaryIds[p] = new int[prior.oneBody[p].length];
            Arrays.fill(unaryIds[p], -1);
            for (int rc = 0; rc < unaryIds[p].length; rc++) {
                if (prior.oneBodyCounts[p][rc] == 0) continue;
                unaryIds[p][rc] = ratios.size();
                ratios.add(shrink.unary / Math.max(1, coverage.unary(p, rc)));
            }
        }
        long[] pairKeys = prior.pairSums.keys();
        Arrays.sort(pairKeys);
        Map<Long, Integer> pairIds = new HashMap<>();
        for (long key : pairKeys) {
            pairIds.put(key, ratios.size());
            ratios.add(shrink.pair / Math.max(1,
                    coverage.pairDistinctContexts.getOrDefault(key, 0)));
        }
        int[][] features = new int[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            int[] conf = samples.get(i).conf;
            int[] active = new int[conf.length + interactionEdges.length];
            int count = 0;
            for (int p = 0; p < conf.length; p++) {
                if (conf[p] >= 0 && unaryIds[p][conf[p]] >= 0)
                    active[count++] = unaryIds[p][conf[p]];
            }
            for (int[] edge : interactionEdges) {
                int a = edge[0], b = edge[1];
                if (conf[a] < 0 || conf[b] < 0) continue;
                Integer id = pairIds.get(EtaCorrections.packPairKey(a, conf[a], b, conf[b]));
                if (id != null) active[count++] = id;
            }
            features[i] = Arrays.copyOf(active, count);
        }
        PackStarProposalLearning.Data data = proposalLearningData(samples, prior);
        double[] regularization = new double[ratios.size()];
        for (int j = 0; j < ratios.size(); j++) regularization[j] = ratios.get(j);
        PackStarProposalLearning.JointFit fit = PackStarProposalLearning.jointFit(
                features, data.residual, data.logWeight, regularization,
                PAIR_ETA_RESIDUAL_CAP_KCAL, 40);
        for (int p = 0; p < unaryIds.length; p++) {
            for (int rc = 0; rc < unaryIds[p].length; rc++) {
                int id = unaryIds[p][rc];
                if (id >= 0) prior.oneBody[p][rc] += fit.delta[id] * prior.oneBodyCounts[p][rc];
            }
        }
        for (long key : pairKeys) {
            int count = prior.pairSums.count(key);
            prior.pairSums.put(key, prior.pairSums.sum(key) + fit.delta[pairIds.get(key)] * count, count);
        }
        prior.globalOffsetKcal += fit.offset;
        System.out.println("[PACK*-joint-pair-fit] samples=" + samples.size()
                + ", observedCells=" + ratios.size() + ", offsetKcal=" + fit.offset);
        return prior;
    }

    private MeanExcessFoldScore scoreFrequencySeverityEtaFold(
            List<CCDResult> samples,
            EtaCorrections eta,
            MeanExcessFoldScratch scratch,
            MeanExcessCoverageFlags coverageFlags,
            MeanExcessEtaEnergyCache energyCache) {
        int n = samples.size();
        if (n <= 1) {
            return invalidFrequencySeverityFoldScore(n);
        }
        scratch.ensureCapacity(n);
        double[] logA = scratch.logA;
        double[] logV = scratch.logV;
        for (int i = 0; i < n; i++) {
            CCDResult sample = samples.get(i);
            double targetEnergy = sample.eMin
                    + energyCache.getEnergy(sample, eta);
            logA[i] = meanExcessSourceLogWeight(
                    sample, targetEnergy);
            logV[i] = logA[i]
                    + (targetEnergy - sample.eTrue) / RT;
            if (!Double.isFinite(logA[i]) || !Double.isFinite(logV[i])) {
                return invalidFrequencySeverityFoldScore(n);
            }
        }

        double logSumA = logSumExp(logA);
        double logSumV = logSumExp(logV);
        if (!Double.isFinite(logSumA) || !Double.isFinite(logSumV)) {
            return invalidFrequencySeverityFoldScore(n);
        }
        double logMu = logSumV - logSumA;
        double[] proposalWeights = scratch.proposalWeights;
        double[] logRelative = scratch.logRelative;
        double sumWeightSquares = 0.0;
        double clippedMean = 0.0;
        double exceedanceProbability = 0.0;
        double candidateUnaryUndertrained = 0.0;
        double candidatePairUndertrained = 0.0;
        int baselineUnaryUndertrained = 0;
        int baselinePairUndertrained = 0;
        double logClip = meanExcessRelativeBoundKcal / RT;
        double[] bulk = scratch.bulk;
        for (int i = 0; i < n; i++) {
            double weight = Math.exp(logA[i] - logSumA);
            double logR = logV[i] - logA[i] - logMu;
            if (!Double.isFinite(weight) || weight < 0.0
                    || !Double.isFinite(logR)) {
                return invalidFrequencySeverityFoldScore(n);
            }
            proposalWeights[i] = weight;
            logRelative[i] = logR;
            sumWeightSquares += weight * weight;
            double value = logR >= logClip
                    ? 1.0 : Math.exp(logR - logClip);
            bulk[i] = value;
            clippedMean += weight * value;
            if (logR > logClip) exceedanceProbability += weight;
            if (coverageFlags.unaryUndertrained[i]) {
                baselineUnaryUndertrained++;
                candidateUnaryUndertrained += weight;
            }
            if (coverageFlags.pairUndertrained[i]) {
                baselinePairUndertrained++;
                candidatePairUndertrained += weight;
            }
        }
        if (!(sumWeightSquares > 0.0)
                || !Double.isFinite(sumWeightSquares)) {
            return invalidFrequencySeverityFoldScore(n);
        }
        double shiftEss = 1.0 / sumWeightSquares;
        double shiftEssFraction = shiftEss / n;
        double varianceDenominator = 1.0 - sumWeightSquares;
        if (!(varianceDenominator > 0.0)) {
            return invalidFrequencySeverityFoldScore(n);
        }
        double clippedVariance = 0.0;
        for (int i = 0; i < n; i++) {
            double centered = bulk[i] - clippedMean;
            clippedVariance += proposalWeights[i] * centered * centered;
        }
        clippedVariance /= varianceDenominator;

        double logMeanR = weightedLogMeanExp(
                proposalWeights, logRelative, 1.0);
        double logSecondR = weightedLogMeanExp(
                proposalWeights, logRelative, 2.0);
        double logTargetM2 = logSecondR - 2.0 * logMeanR;
        double targetM2 = logTargetM2 < Math.log(Double.MAX_VALUE)
                ? Math.max(1.0, Math.exp(logTargetM2))
                : Double.POSITIVE_INFINITY;
        double baselineUnaryMass =
                (double) baselineUnaryUndertrained / n;
        double baselinePairMass =
                (double) baselinePairUndertrained / n;
        double unaryAmplification = massAmplification(
                candidateUnaryUndertrained, baselineUnaryMass);
        double pairAmplification = massAmplification(
                candidatePairUndertrained, baselinePairMass);

        try {
            // Preserve parameter validation without sizing individual folds.
            PackStarFrequencySeverityPAC.validateSizingInputs(
                    new PackStarFrequencySeverityPAC.Moments(
                            clamp(clippedMean, 0.0, 1.0),
                            Math.max(0.0, clippedVariance) * nstarInflate,
                            clamp(exceedanceProbability, 0.0, 1.0)),
                    maxEstSamples,
                    configuredFrequencySeverityUnreachableSamples(),
                    targetEpsilon, meanExcessSizeSafety, conditionalMeanExcessBound,
                    meanExcessEventDelta(), meanExcessEventDelta());
        } catch (IllegalArgumentException ex) {
            return invalidFrequencySeverityFoldScore(n);
        }

        boolean finite = Double.isFinite(logMu)
                && Double.isFinite(shiftEssFraction)
                && Double.isFinite(logTargetM2)
                && Double.isFinite(clippedMean)
                && Double.isFinite(clippedVariance)
                && Double.isFinite(exceedanceProbability)
                && Double.isFinite(unaryAmplification)
                && Double.isFinite(pairAmplification);
        boolean shiftPass = shiftEssFraction + 1.0e-12
                >= meanExcessMinShiftEssFraction;
        return new MeanExcessFoldScore(
                n, logMu, shiftEss,
                shiftEssFraction, targetM2,
                clippedMean, clippedVariance, exceedanceProbability,
                baselineUnaryMass, candidateUnaryUndertrained,
                baselinePairMass,
                candidatePairUndertrained, finite, shiftPass);
    }

    private MeanExcessFoldScore invalidFrequencySeverityFoldScore(int sampleCount) {
        PackStarFrequencySeverityPAC.validateSizingInputs(
                new PackStarFrequencySeverityPAC.Moments(0.0, 0.0, 1.0),
                maxEstSamples, configuredFrequencySeverityUnreachableSamples(),
                targetEpsilon, meanExcessSizeSafety, conditionalMeanExcessBound,
                meanExcessEventDelta(), meanExcessEventDelta());
        return new MeanExcessFoldScore(
                sampleCount, Double.NaN, 0.0, 0.0,
                Double.POSITIVE_INFINITY, 0.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0, false, false);
    }

    private MeanExcessCrossfitScore poolFrequencySeverityFoldScores(
            MeanExcessFoldScore[] folds) {
        if (folds == null || folds.length == 0) {
            return invalidFrequencySeverityCrossfitScore();
        }
        int[] sampleCounts = new int[folds.length];
        double[] shiftEss = new double[folds.length];
        double[] clippedMeans = new double[folds.length];
        double[] clippedVariances = new double[folds.length];
        double[] exceedanceProbabilities = new double[folds.length];
        int total = 0;
        boolean allFoldsFinite = true;
        boolean allFoldsShiftPass = true;
        for (int fold = 0; fold < folds.length; fold++) {
            MeanExcessFoldScore score = folds[fold];
            if (score == null) return invalidFrequencySeverityCrossfitScore();
            sampleCounts[fold] = score.sampleCount;
            shiftEss[fold] = score.shiftEss;
            clippedMeans[fold] = score.clippedMean;
            clippedVariances[fold] = score.clippedVariance;
            exceedanceProbabilities[fold] = score.exceedanceProbability;
            total += score.sampleCount;
            allFoldsFinite &= score.finite;
            allFoldsShiftPass &= score.shiftPass;
        }
        if (!allFoldsFinite || total <= 1) {
            return invalidFrequencySeverityCrossfitScore();
        }

        PackStarFrequencySeverityPAC.PooledCrossfitMoments pooled;
        try {
            pooled = PackStarFrequencySeverityPAC.poolCrossfitMoments(
                    sampleCounts, shiftEss, clippedMeans,
                    clippedVariances, exceedanceProbabilities);
        } catch (IllegalArgumentException ex) {
            return invalidFrequencySeverityCrossfitScore();
        }

        double logMu = 0.0;
        double targetM2 = 0.0;
        double baselineUnaryMass = 0.0;
        double candidateUnaryMass = 0.0;
        double baselinePairMass = 0.0;
        double candidatePairMass = 0.0;
        for (MeanExcessFoldScore fold : folds) {
            double foldMass = (double) fold.sampleCount / total;
            logMu += foldMass * fold.logMu;
            targetM2 += foldMass * fold.targetM2;
            baselineUnaryMass += foldMass
                    * fold.baselineUnaryUndertrainedMass;
            candidateUnaryMass += foldMass
                    * fold.candidateUnaryUndertrainedMass;
            baselinePairMass += foldMass
                    * fold.baselinePairUndertrainedMass;
            candidatePairMass += foldMass
                    * fold.candidatePairUndertrainedMass;
        }
        double unaryAmplification = massAmplification(
                candidateUnaryMass, baselineUnaryMass);
        double pairAmplification = massAmplification(
                candidatePairMass, baselinePairMass);
        double targetEssFraction = Double.isFinite(targetM2)
                && targetM2 > 0.0 ? 1.0 / targetM2 : 0.0;

        PackStarFrequencySeverityPAC.Sizing sizing;
        PackStarFrequencySeverityPAC.Interval atMax;
        try {
            PackStarFrequencySeverityPAC.Moments inflated =
                    new PackStarFrequencySeverityPAC.Moments(
                            pooled.moments.clippedMean,
                            pooled.moments.clippedVariance * nstarInflate,
                            pooled.moments.exceedanceProbability);
            sizing = PackStarFrequencySeverityPAC.size(
                    inflated, maxEstSamples,
                    configuredFrequencySeverityUnreachableSamples(),
                    targetEpsilon, meanExcessSizeSafety,
                    conditionalMeanExcessBound, meanExcessEventDelta(),
                    meanExcessEventDelta());
            atMax = PackStarFrequencySeverityPAC.project(
                    maxEstSamples, inflated, conditionalMeanExcessBound,
                    meanExcessEventDelta(),
                    meanExcessEventDelta());
        } catch (IllegalArgumentException ex) {
            return invalidFrequencySeverityCrossfitScore();
        }

        boolean finite = Double.isFinite(logMu)
                && Double.isFinite(pooled.effectiveSampleSize)
                && Double.isFinite(pooled.effectiveSampleFraction)
                && Double.isFinite(targetEssFraction)
                && Double.isFinite(pooled.moments.clippedMean)
                && Double.isFinite(pooled.moments.clippedVariance)
                && Double.isFinite(pooled.moments.exceedanceProbability)
                && Double.isFinite(unaryAmplification)
                && Double.isFinite(pairAmplification);
        boolean shiftPass = finite
                && pooled.effectiveSampleFraction + 1.0e-12
                >= meanExcessMinShiftEssFraction;
        boolean bulkPass = atMax.clippedLower > 0.0;
        boolean reachabilityPass = sizing.reachableAtMax
                && sizing.epsilonAtMaxSamples
                <= targetEpsilon + 1.0e-12;
        return new MeanExcessCrossfitScore(
                pooled.effectiveSampleFraction,
                targetM2, unaryAmplification,
                pairAmplification, sizing,
                allFoldsFinite, allFoldsShiftPass,
                finite, shiftPass,
                bulkPass, reachabilityPass);
    }

    private MeanExcessCrossfitScore invalidFrequencySeverityCrossfitScore() {
        PackStarFrequencySeverityPAC.Moments fallback =
                new PackStarFrequencySeverityPAC.Moments(0.0, 0.0, 1.0);
        PackStarFrequencySeverityPAC.Sizing sizing =
                PackStarFrequencySeverityPAC.size(
                        fallback, maxEstSamples,
                        configuredFrequencySeverityUnreachableSamples(),
                        targetEpsilon, meanExcessSizeSafety,
                        conditionalMeanExcessBound, meanExcessEventDelta(),
                        meanExcessEventDelta());
        return new MeanExcessCrossfitScore(
                0.0,
                Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY,
                sizing, false, false,
                false, false, false, false);
    }

    private MeanExcessCandidateScore selectBestFrequencySeverityCandidate(
            List<MeanExcessCandidateScore> candidates) {
        List<MeanExcessCandidateScore> eligible = new ArrayList<>();
        for (MeanExcessCandidateScore candidate : candidates)
            if (candidate.eligible) eligible.add(candidate);
        return PackStarProposalLearning.chooseWithinFivePercent(eligible,
                candidate -> candidate.crossfit.sizing.finalSamples,
                (a, b) -> {
                    if (isBetterFrequencySeverityCandidate(a, b)) return -1;
                    if (isBetterFrequencySeverityCandidate(b, a)) return 1;
                    return 0;
                });
    }

    /**
     * Adaptation-only trust-region probe.  If no proposal is ready to issue a
     * bound, use the strongest unshrunk pair-eta step that still retains at
     * least the pre-registered discovery ESS in every fold and in the full
     * refit. Rank these graph-preserving proposals by their off-policy projection.
     * Reachability and sparse-cell coverage are deliberately not required:
     * the probe exists to obtain the on-policy data those q_m diagnostics
     * cannot supply, and it can never reach final sampling without a fresh
     * on-policy validation batch.
     */
    private MeanExcessCandidateScore selectFrequencySeverityDiscoveryProbe(
            List<MeanExcessCandidateScore> candidates) {
        MeanExcessCandidateScore strongest = null;
        for (MeanExcessCandidateScore candidate : candidates) {
            if (candidate.shrink.unary != 0.0
                    || candidate.shrink.pair != 0.0
                    || candidate.crossfit == null
                    || candidate.refitAudit == null
                    || !candidate.crossfit.finite
                    || !candidate.crossfit.allFoldsFinite
                    || !candidate.refitAudit.finite
                    || candidate.crossfit.shiftEssFraction + 1.0e-12
                    < meanExcessDiscoveryMinShiftEssFraction
                    || candidate.refitAudit.shiftEssFraction + 1.0e-12
                    < meanExcessDiscoveryMinShiftEssFraction) {
                continue;
            }
            boolean everyFoldRetainsDiscoveryOverlap = true;
            for (MeanExcessFoldScore fold : candidate.folds) {
                if (!fold.finite
                        || fold.shiftEssFraction + 1.0e-12
                        < meanExcessDiscoveryMinShiftEssFraction) {
                    everyFoldRetainsDiscoveryOverlap = false;
                    break;
                }
            }
            if (!everyFoldRetainsDiscoveryOverlap) continue;
            if (strongest == null
                    || isBetterFrequencySeverityDiscoveryStructure(
                    candidate, strongest)) {
                strongest = candidate;
            }
        }
        return strongest;
    }

    private static boolean isBetterFrequencySeverityDiscoveryStructure(
            MeanExcessCandidateScore candidate,
            MeanExcessCandidateScore incumbent) {
        int compare = Double.compare(
                candidate.crossfit.sizing.epsilonAtMaxSamples,
                incumbent.crossfit.sizing.epsilonAtMaxSamples);
        if (compare != 0) return compare < 0;
        compare = Double.compare(candidate.crossfit.targetM2,
                incumbent.crossfit.targetM2);
        if (compare != 0) return compare < 0;
        compare = Double.compare(
                candidate.crossfit.shiftEssFraction,
                incumbent.crossfit.shiftEssFraction);
        if (compare != 0) return compare > 0;
        return candidate.id.compareTo(incumbent.id) < 0;
    }

    private boolean meanExcessPilotPass(
            PackStarFrequencySeverityPAC.Interval interval,
            PackStarFrequencySeverityPAC.Sizing sizing) {
        return interval != null && sizing != null
                && interval.hasPositiveBulkLower()
                && sizing.reachableAtMax
                && sizing.epsilonAtMaxSamples
                <= targetEpsilon + 1.0e-12;
    }

    /**
     * Decide whether a failed, adaptation-only discovery pilot is promising
     * enough to receive more IID samples from the exact same frozen proposal.
     * This rule never changes eta and never consumes validation or final data.
     */
    static boolean shouldExtendFrequencySeverityDiscovery(
            int currentSamples, int maximumSamples,
            PackStarFrequencySeverityPAC.Interval interval,
            PackStarFrequencySeverityPAC.Sizing sizing,
            double targetEpsilon) {
        return currentSamples >= 2
                && currentSamples < maximumSamples
                && interval != null
                && interval.isFinite()
                && !interval.hasPositiveBulkLower()
                && sizing != null
                && sizing.reachableAtMax
                && Double.isFinite(sizing.epsilonAtMaxSamples)
                && Double.isFinite(targetEpsilon)
                && sizing.epsilonAtMaxSamples
                <= targetEpsilon + 1.0e-12;
    }

    private MeanExcessPilotEvaluation evaluateFrequencySeverityPilot(
            List<CCDResult> samples, EtaCorrections eta,
            double logMuTrain) {
        double[] logRelative = computeFrequencySeverityLogRelativeWeights(
                samples, eta, logMuTrain);
        PackStarFrequencySeverityPAC.Interval interval =
                PackStarFrequencySeverityPAC.evaluate(
                        logRelative,
                        meanExcessRelativeBoundKcal / RT,
                        conditionalMeanExcessBound,
                        meanExcessEventDelta(),
                        meanExcessEventDelta());
        PackStarFrequencySeverityPAC.Moments sizingMoments =
                new PackStarFrequencySeverityPAC.Moments(
                        interval.clippedMean,
                        interval.clippedVariance * nstarInflate,
                        interval.exceedanceProbabilityEmpirical);
        PackStarFrequencySeverityPAC.Sizing sizing =
                PackStarFrequencySeverityPAC.size(
                        sizingMoments, maxEstSamples,
                        configuredFrequencySeverityUnreachableSamples(),
                        targetEpsilon, meanExcessSizeSafety,
                        conditionalMeanExcessBound,
                        meanExcessEventDelta(),
                        meanExcessEventDelta());
        return new MeanExcessPilotEvaluation(
                interval, sizing);
    }

    private static boolean isBetterFrequencySeverityCandidate(
            MeanExcessCandidateScore candidate,
            MeanExcessCandidateScore incumbent) {
        MeanExcessCrossfitScore candidateScore = candidate.crossfit;
        MeanExcessCrossfitScore incumbentScore = incumbent.crossfit;
        if (candidateScore.sizing.finalSamples
                != incumbentScore.sizing.finalSamples) {
            return candidateScore.sizing.finalSamples
                    < incumbentScore.sizing.finalSamples;
        }
        int compare = Double.compare(
                candidateScore.sizing.epsilonAtMaxSamples,
                incumbentScore.sizing.epsilonAtMaxSamples);
        if (compare != 0) return compare < 0;
        compare = Double.compare(candidateScore.targetM2,
                incumbentScore.targetM2);
        if (compare != 0) return compare < 0;
        compare = Double.compare(
                Math.max(candidateScore.unaryUndertrainedAmplification,
                        candidateScore.pairUndertrainedAmplification),
                Math.max(incumbentScore.unaryUndertrainedAmplification,
                        incumbentScore.pairUndertrainedAmplification));
        if (compare != 0) return compare < 0;
        return candidate.id.compareTo(incumbent.id) < 0;
    }

    private int configuredFrequencySeverityUnreachableSamples() {
        int configured = getConfigInteger(
                "packstar.pac.unreachableCap",
                Math.min(maxEstSamples, Math.max(2 * pilotSamples, 400)));
        return Math.max(2, Math.min(maxEstSamples, configured));
    }

    /** Two confidence events for each of the three K* partition functions. */
    private double meanExcessEventDelta() {
        return delta / 6.0;
    }

    private double massAmplification(double candidateMass,
                                     double baselineMass) {
        if (baselineMass <= 1.0e-15) {
            return candidateMass <= 1.0e-15
                    ? 1.0 : Double.POSITIVE_INFINITY;
        }
        return candidateMass / baselineMass;
    }

    private MeanExcessRefitAudit auditFrequencySeverityRefit(
            List<CCDResult> samples, EtaCorrections eta,
            MeanExcessCoverageFlags coverageFlags,
            MeanExcessFoldScratch scratch,
            MeanExcessEtaEnergyCache energyCache) {
        int n = samples.size();
        scratch.ensureCapacity(n);
        double[] logA = scratch.logA;
        double logSumA = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            CCDResult sample = samples.get(i);
            double targetEnergy = sample.eMin
                    + energyCache.getEnergy(sample, eta);
            logA[i] = meanExcessSourceLogWeight(
                    sample, targetEnergy);
            logSumA = logAddExp(logSumA, logA[i]);
        }
        if (!Double.isFinite(logSumA)) {
            return new MeanExcessRefitAudit(
                    Double.NaN, false, false, false);
        }

        int baselineUnaryCount = 0;
        int baselinePairCount = 0;
        double candidateUnaryMass = 0.0;
        double candidatePairMass = 0.0;
        double sumWeightSquares = 0.0;
        for (int i = 0; i < n; i++) {
            double weight = Math.exp(logA[i] - logSumA);
            sumWeightSquares += weight * weight;
            if (coverageFlags.unaryUndertrained[i]) {
                baselineUnaryCount++;
                candidateUnaryMass += weight;
            }
            if (coverageFlags.pairUndertrained[i]) {
                baselinePairCount++;
                candidatePairMass += weight;
            }
        }
        double shiftEssFraction = sumWeightSquares > 0.0
                && Double.isFinite(sumWeightSquares)
                ? 1.0 / (n * sumWeightSquares) : Double.NaN;
        double baselineUnaryMass = (double) baselineUnaryCount / n;
        double baselinePairMass = (double) baselinePairCount / n;
        double unaryAmplification = massAmplification(
                candidateUnaryMass, baselineUnaryMass);
        double pairAmplification = massAmplification(
                candidatePairMass, baselinePairMass);
        boolean finite = Double.isFinite(shiftEssFraction)
                && Double.isFinite(candidateUnaryMass)
                && Double.isFinite(candidatePairMass)
                && Double.isFinite(unaryAmplification)
                && Double.isFinite(pairAmplification);
        boolean shiftPass = finite && shiftEssFraction + 1.0e-12
                >= meanExcessMinShiftEssFraction;
        boolean coveragePass = finite
                && unaryAmplification
                <= meanExcessMaxUndertrainedAmplification + 1.0e-12
                && pairAmplification
                <= meanExcessMaxUndertrainedAmplification + 1.0e-12;
        return new MeanExcessRefitAudit(
                shiftEssFraction,
                finite, shiftPass, coveragePass);
    }

    private double weightedLogMeanExp(double[] normalizedWeights,
                                      double[] logValues,
                                      double power) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < normalizedWeights.length; i++) {
            if (!(normalizedWeights[i] > 0.0)) continue;
            max = Math.max(max,
                    Math.log(normalizedWeights[i]) + power * logValues[i]);
        }
        if (!Double.isFinite(max)) return Double.NaN;
        double sum = 0.0;
        for (int i = 0; i < normalizedWeights.length; i++) {
            if (!(normalizedWeights[i] > 0.0)) continue;
            sum += Math.exp(Math.log(normalizedWeights[i])
                    + power * logValues[i] - max);
        }
        return max + Math.log(sum);
    }

    private static double logSumExp(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) max = Math.max(max, value);
        if (!Double.isFinite(max)) return max;
        double sum = 0.0;
        for (double value : values) sum += Math.exp(value - max);
        return max + Math.log(sum);
    }

    /**
     * Unnormalized log(q_target/q_source).  The omitted -logZ_target is
     * common to every adaptation sample and therefore cancels under
     * self-normalization.  Retaining logZ_source is essential when batches
     * came from different sequential proposals.
     */
    static double meanExcessSourceLogWeight(
            double targetEnergy, double sourceEnergy,
            double sourceLogZ, double rt) {
        if (!Double.isFinite(targetEnergy)
                || !Double.isFinite(sourceEnergy)
                || !Double.isFinite(sourceLogZ)
                || !Double.isFinite(rt) || !(rt > 0.0)) {
            return Double.NaN;
        }
        return -(targetEnergy - sourceEnergy) / rt + sourceLogZ;
    }

    private double meanExcessSourceLogWeight(
            CCDResult sample, double targetEnergy) {
        if (Double.isFinite(sample.sourceProposalEnergy)
                && Double.isFinite(sample.sourceProposalLogZ)
                && sample.sourceProposalId != null
                && !sample.sourceProposalId.isEmpty()) {
            return meanExcessSourceLogWeight(
                    targetEnergy, sample.sourceProposalEnergy,
                    sample.sourceProposalLogZ, RT);
        }
        // Every sample entering candidate fitting must carry its exact source
        // proposal.  Returning NaN fails the candidate closed instead of
        // silently mixing the production route with an obsolete q_m fallback.
        return Double.NaN;
    }

    private double computeEtaEnergy(CCDResult sample, EtaCorrections eta) {
        int[] conf = sample.conf;
        long[] pairKeys = sample.features == null
                ? null : sample.features.pairKeys;
        double energy = eta.globalOffsetKcal;
        for (int pos = 0; pos < conf.length; pos++) {
            int rc = conf[pos];
            if (rc < 0) continue;
            energy += eta.getOneBodyEta(pos, rc);
            for (int edgeIndex : interactionEdgeIndicesByFirstPosition[pos]) {
                int pos2 = interactionEdges[edgeIndex][1];
                int rc2 = conf[pos2];
                if (rc2 < 0) continue;
                long pairKey = pairKeys == null
                        ? EtaCorrections.packPairKey(pos, rc, pos2, rc2)
                        : pairKeys[edgeIndex];
                energy += eta.getPairEta(pairKey);
            }
        }
        return energy;
    }

    private boolean touchesFrequencySeverityUndertrainedUnary(
            int[] conf, MeanExcessEtaCoverage coverage) {
        return touchesFrequencySeverityUnaryBelowCount(
                conf, coverage, meanExcessMinTrainCount);
    }

    private boolean touchesFrequencySeverityUnaryBelowCount(
            int[] conf, MeanExcessEtaCoverage coverage,
            int minimumCount) {
        for (int pos = 0; pos < conf.length; pos++) {
            int rc = conf[pos];
            if (rc >= 0 && coverage.unary(pos, rc) < minimumCount) {
                return true;
            }
        }
        return false;
    }

    private boolean touchesFrequencySeverityUndertrainedPair(
            int[] conf, MeanExcessEtaCoverage coverage) {
        return touchesFrequencySeverityPairBelowCount(
                conf, coverage, meanExcessMinTrainCount);
    }

    private boolean touchesFrequencySeverityPairBelowCount(
            int[] conf, MeanExcessEtaCoverage coverage,
            int minimumCount) {
        for (int[] edge : interactionEdges) {
            int pos1 = edge[0];
            int pos2 = edge[1];
            int rc1 = conf[pos1];
            int rc2 = conf[pos2];
            if (rc1 < 0 || rc2 < 0) continue;
            if (coverage.pair(pos1, rc1, pos2, rc2)
                    < minimumCount) return true;
        }
        return false;
    }

    private EtaCorrections shrinkEta(EtaCorrections raw,
                                     MeanExcessEtaCoverage coverage,
                                     double unaryShrink,
                                     double pairShrink) {
        int[] numRCs = new int[raw.oneBody.length];
        for (int pos = 0; pos < raw.oneBody.length; pos++) {
            numRCs[pos] = raw.oneBody[pos].length;
        }
        EtaCorrections shrunk = new EtaCorrections(
                raw.oneBody.length, numRCs, raw.pairSums.size());
        for (int pos = 0; pos < raw.oneBody.length; pos++) {
            for (int rc = 0; rc < raw.oneBody[pos].length; rc++) {
                int rawCount = raw.oneBodyCounts[pos][rc];
                if (rawCount <= 0) continue;
                int contexts = coverage.unary(pos, rc);
                double factor = unaryShrink == 0.0
                        ? 1.0 : contexts / (contexts + unaryShrink);
                double value = factor * raw.getOneBodyEta(pos, rc);
                shrunk.oneBody[pos][rc] = value * rawCount;
                shrunk.oneBodyCounts[pos][rc] = rawCount;
                shrunk.oneBodyCount++;
            }
        }
        raw.pairSums.forEach((key, sum, count) -> {
            if (count <= 0) return;
            int contexts = coverage.pairDistinctContexts.getOrDefault(
                    key, 0);
            double factor = pairShrink == 0.0
                    ? 1.0 : contexts / (contexts + pairShrink);
            shrunk.pairSums.put(key, factor * sum, count);
        });
        shrunk.globalOffsetKcal = raw.globalOffsetKcal;
        return shrunk;
    }

    private void runAdaptiveFrequencySeverityPAC(long startTime) {
        double initialProposalLogZ = rootedRootEdge.getLogZUpper(0);
        if (!Double.isFinite(initialProposalLogZ)) {

            failCertificate("AdaptiveFrequencySeverityPAC: initial q_m proposal"
                    + " normalizer is non-finite", startTime);
            return;
        }

        System.out.println("[PACK*-adaptive-frequency-severity] enabled"
                + "; proposalLearning=true"
                + "; qMTrain=" + trainSamples
                + "; discoveryPerRound=" + meanExcessDiscoverySamples
                + "->" + meanExcessDiscoveryMaxSamples
                + "; maxRefits=" + meanExcessMaxRefits
                + "; validation=" + meanExcessValidationSamples
                + "; folds=" + MEAN_EXCESS_FOLDS
                + "; finalMinShiftESS=" + meanExcessMinShiftEssFraction
                + "; discoveryMinShiftESS="
                + meanExcessDiscoveryMinShiftEssFraction
                + "; candidateDP=0; proposalDP<="
                + (meanExcessMaxRefits + 1)
                + "; alpha=1-fixed; proposal=unary-pair");

        String trainStage = "adaptive-frequency-severity-qm-train";
        List<CCDResult> trainCCD = runParallelCCD(
                sampleConformationsFromDP(
                        trainSamples, stageRandom(trainStage)),
                branchMinimizingEmat, initialProposalLogZ,
                "adaptive-frequency-severity-qm");
        boolean qMLowerBoundValid = validateObservedLowerBound(
                trainCCD, trainStage);
        if (trainCCD.size() != trainSamples || !qMLowerBoundValid) {

            failCertificate("AdaptiveFrequencySeverityPAC: invalid or incomplete"
                    + " q_m training sample", startTime);
            return;
        }

        List<CCDResult> fitPool = new ArrayList<>(trainCCD);
        List<CCDResult> history = Collections.emptyList();
        List<CCDResult> scoringBatch = trainCCD;
        int discoveryCCDCalls = 0;
        int proposalDpSweeps = 0;

        MeanExcessCandidateScore selected = null;
        EtaCorrections selectedEta = null;
        EnergyMatrix selectedEmat = null;
        double selectedLogZ = Double.NaN;
        double selectedLogMu = Double.NaN;
        int selectedRound = -1;
        boolean selectedWasDiscoveryProbe = false;

        for (int round = 0; round <= meanExcessMaxRefits; round++) {
            MeanExcessEtaTraining training;
            List<MeanExcessCandidateScore> candidates;
            MeanExcessCandidateScore finalEligible;
            MeanExcessCandidateScore discoveryProbe;
            try {
                training = extractFrequencySeverityEtaTraining(
                        history, scoringBatch);
                candidates = scoreFrequencySeverityEtaCandidates(
                        scoringBatch, training);
                finalEligible = selectBestFrequencySeverityCandidate(candidates);
                discoveryProbe = finalEligible == null
                        ? selectFrequencySeverityDiscoveryProbe(candidates)
                        : null;
            } catch (Exception ex) {

                failCertificate("AdaptiveFrequencySeverityPAC round " + round
                        + " eta fit/score failed: "
                        + sanitizeTsv(ex.getMessage()), startTime);
                return;
            }

            MeanExcessCandidateScore roundSelection =
                    finalEligible != null ? finalEligible : discoveryProbe;
            boolean roundUsesDiscoveryProbe = finalEligible == null;
            if (roundSelection == null) {

                failCertificate("AdaptiveFrequencySeverityPAC: no final candidate"
                        + " and no discovery probe retained source-shift ESS >= "
                        + meanExcessDiscoveryMinShiftEssFraction,
                        startTime);
                return;
            }

            EtaCorrections roundEta = roundSelection.fullEta;
            double roundLogMu = computeFrequencySeverityTrainingLogMu(
                    fitPool, roundEta);
            if (!Double.isFinite(roundLogMu)) {

                failCertificate("AdaptiveFrequencySeverityPAC: source-aware"
                        + " training gauge is non-finite", startTime);
                return;
            }

            EnergyMatrix roundEmat = buildCorrectedEmat(roundEta);
            CorrectedDPResult correctedDP = recomputeDP(roundEmat);
            proposalDpSweeps++;
            double roundLogZ = correctedDP.logZCorrected;
            if (!Double.isFinite(roundLogZ)) {

                failCertificate("AdaptiveFrequencySeverityPAC round " + round
                        + ": proposal DP normalizer is non-finite", startTime);
                return;
            }

            String discoveryStage = String.format(
                    Locale.ROOT,
                    "adaptive-frequency-severity-discovery-%02d", round);
            List<CCDResult> discoverySamples = new ArrayList<>(runParallelCCD(
                    sampleConformationsFromDP(
                            meanExcessDiscoverySamples,
                            stageRandom(discoveryStage)),
                    roundEmat, roundLogZ,
                    "adaptive-frequency-severity-proposal-" + round));
            discoveryCCDCalls += discoverySamples.size();
            if (discoverySamples.size() != meanExcessDiscoverySamples
                    || !validateObservedLowerBound(
                    discoverySamples, discoveryStage)) {

                failCertificate("AdaptiveFrequencySeverityPAC round " + round
                        + ": invalid or incomplete discovery sample", startTime);
                return;
            }

            MeanExcessPilotEvaluation initialEvaluation;
            try {
                initialEvaluation = evaluateFrequencySeverityPilot(
                        discoverySamples, roundEta, roundLogMu);
            } catch (RuntimeException ex) {

                failCertificate("AdaptiveFrequencySeverityPAC round " + round
                        + " initial discovery evaluation failed: "
                        + sanitizeTsv(ex.getMessage()), startTime);
                return;
            }

            boolean discoveryExtended =
                    shouldExtendFrequencySeverityDiscovery(
                            discoverySamples.size(),
                            meanExcessDiscoveryMaxSamples,
                            initialEvaluation.interval,
                            initialEvaluation.sizing,
                            targetEpsilon);
            if (discoveryExtended) {

                int requestedExtension = meanExcessDiscoveryMaxSamples
                        - discoverySamples.size();
                String extensionStage = discoveryStage + "-extension";
                System.out.println("[PACK*-adaptive-frequency-severity] round="
                        + round
                        + ", initialDiscovery=" + discoverySamples.size()
                        + ", initialTail=" + initialEvaluation.interval.exceedanceCount
                        + ", initialBulkLower="
                        + String.format(Locale.ROOT, "%.9g",
                        initialEvaluation.interval.clippedLower)
                        + ", initialEpsilonAtMax="
                        + String.format(Locale.ROOT, "%.9f",
                        initialEvaluation.sizing.epsilonAtMaxSamples)
                        + ", action=extend-same-frozen-proposal-to-"
                        + meanExcessDiscoveryMaxSamples);
                List<CCDResult> extensionSamples = runParallelCCD(
                        sampleConformationsFromDP(
                                requestedExtension,
                                stageRandom(extensionStage)),
                        roundEmat, roundLogZ,
                        "adaptive-frequency-severity-proposal-" + round);
                int extensionDiscoveryCount = extensionSamples.size();
                discoveryCCDCalls += extensionDiscoveryCount;
                if (extensionDiscoveryCount != requestedExtension
                        || !validateObservedLowerBound(
                        extensionSamples, extensionStage)) {

                    failCertificate("AdaptiveFrequencySeverityPAC round " + round
                            + ": invalid or incomplete same-proposal discovery"
                            + " extension", startTime);
                    return;
                }
                discoverySamples.addAll(extensionSamples);
            }

            MeanExcessPilotEvaluation roundEvaluation;
            boolean preliminaryPass;
            try {
                roundEvaluation = discoveryExtended
                        ? evaluateFrequencySeverityPilot(
                        discoverySamples, roundEta, roundLogMu)
                        : initialEvaluation;
                preliminaryPass = meanExcessPilotPass(
                        roundEvaluation.interval, roundEvaluation.sizing);

            } catch (RuntimeException ex) {

                failCertificate("AdaptiveFrequencySeverityPAC round " + round
                        + " discovery evaluation failed: "
                        + sanitizeTsv(ex.getMessage()), startTime);
                return;
            }

            System.out.println("[PACK*-adaptive-frequency-severity] round=" + round
                    + ", role=" + (roundUsesDiscoveryProbe
                    ? "discovery-probe" : "final-eligible")
                    + ", candidate=" + roundSelection.id
                    + ", fitPool=" + fitPool.size()
                    + ", sameProposalExtended=" + discoveryExtended
                    + ", tail=" + roundEvaluation.interval.exceedanceCount
                    + "/" + roundEvaluation.interval.sampleCount
                    + ", bulkLower="
                    + String.format(Locale.ROOT, "%.9g",
                    roundEvaluation.interval.clippedLower)
                    + ", reachable=" + roundEvaluation.sizing.reachableAtMax
                    + ", epsilonAtMax="
                    + String.format(Locale.ROOT, "%.9f",
                    roundEvaluation.sizing.epsilonAtMaxSamples)
                    + ", action=" + (preliminaryPass
                    ? "validate" : round < meanExcessMaxRefits
                    ? "merge-and-refit" : "abort"));

            if (preliminaryPass) {
                selected = roundSelection;
                selectedEta = roundEta;
                selectedEmat = roundEmat;
                selectedLogZ = roundLogZ;
                selectedLogMu = roundLogMu;
                selectedRound = round;
                selectedWasDiscoveryProbe = roundUsesDiscoveryProbe;
                break;
            }

            if (round == meanExcessMaxRefits) {

                failCertificate("AdaptiveFrequencySeverityPAC: discovery pilot"
                        + " remained unreachable or bulk-collapsed after "
                        + meanExcessMaxRefits + " refits", startTime);
                return;
            }

            history = new ArrayList<>(fitPool);
            scoringBatch = discoverySamples;
            fitPool.addAll(discoverySamples);
        }

        if (selected == null || selectedEta == null || selectedEmat == null) {
            failCertificate("AdaptiveFrequencySeverityPAC: internal proposal-selection"
                    + " state is incomplete", startTime);
            return;
        }

        String validationStage = "adaptive-frequency-severity-validation";
        List<CCDResult> validationCCD = runParallelCCD(
                sampleConformationsFromDP(
                        meanExcessValidationSamples,
                        stageRandom(validationStage)),
                null, Double.NaN, null);
        if (validationCCD.size() != meanExcessValidationSamples
                || !validateObservedLowerBound(
                validationCCD, validationStage)) {

            failCertificate("AdaptiveFrequencySeverityPAC: invalid or incomplete"
                    + " independent validation", startTime);
            return;
        }

        MeanExcessPilotEvaluation validation;
        try {
            validation = evaluateFrequencySeverityPilot(
                    validationCCD, selectedEta, selectedLogMu);

        } catch (RuntimeException ex) {
            failCertificate("AdaptiveFrequencySeverityPAC validation statistics"
                    + " failed: " + sanitizeTsv(ex.getMessage()), startTime);
            return;
        }
        if (!meanExcessPilotPass(
                validation.interval, validation.sizing)) {

            failCertificate("AdaptiveFrequencySeverityPAC: independent validation"
                    + " did not confirm a reachable positive-bulk proposal", startTime);
            return;
        }

        System.out.println("[PACK*-adaptive-frequency-severity] frozen proposal:"
                + " round=" + selectedRound
                + ", candidate=" + selected.id
                + ", selectedVia=" + (selectedWasDiscoveryProbe
                ? "on-policy-validated-discovery-probe" : "final-eligible-grid")
                + ", fitPool=" + fitPool.size()
                + ", discoveryCCD=" + discoveryCCDCalls
                + ", validationTail=" + validation.interval.exceedanceCount
                + "/" + validation.interval.sampleCount
                + ", validationEpsilonAtMax="
                + String.format(Locale.ROOT, "%.9f",
                validation.sizing.epsilonAtMaxSamples)
                + ", finalN=" + validation.sizing.finalSamples
                + ", proposalDpSweeps=" + proposalDpSweeps);

        String finalStage = "adaptive-frequency-severity-final";
        List<CCDResult> finalCCD = runParallelCCD(
                sampleConformationsFromDP(
                        validation.sizing.finalSamples,
                        stageRandom(finalStage)),
                null, Double.NaN, null);
        if (finalCCD.size() != validation.sizing.finalSamples
                || !validateObservedLowerBound(finalCCD, finalStage)) {

            failCertificate("AdaptiveFrequencySeverityPAC: invalid or incomplete"
                    + " fresh final sample", startTime);
            return;
        }
        double[] finalLogR = computeFrequencySeverityLogRelativeWeights(
                finalCCD, selectedEta, selectedLogMu);

        PackStarFrequencySeverityPAC.Interval finalInterval;
        PackStarFrequencySeverityPAC.MeanExcessTest finalMeanExcessTest;
        try {
            finalInterval = PackStarFrequencySeverityPAC.evaluate(
                    finalLogR, meanExcessRelativeBoundKcal / RT,
                    conditionalMeanExcessBound, meanExcessEventDelta(),
                    meanExcessEventDelta());
            // One fixed-time test on the same fresh batch as the interval.
            // No additional samples, repeated looks, or post-final adaptation.
            finalMeanExcessTest = PackStarFrequencySeverityPAC.testConditionalMeanExcess(
                    finalLogR, meanExcessRelativeBoundKcal / RT,
                    conditionalMeanExcessBound, meanExcessTestAlpha);

        } catch (RuntimeException ex) {
            failCertificate("AdaptiveFrequencySeverityPAC final statistics failed: "
                    + sanitizeTsv(ex.getMessage()), startTime);
            return;
        }
        double logScale = selectedLogZ + selectedLogMu
                + meanExcessRelativeBoundKcal / RT;
        logZLowerPAC = finalInterval.normalizedMeanLower > 0.0
                ? logScale + Math.log(finalInterval.normalizedMeanLower)
                : Double.NEGATIVE_INFINITY;
        logZUpperPAC = finalInterval.normalizedMeanUpper > 0.0
                ? logScale + Math.log(finalInterval.normalizedMeanUpper)
                : Double.NaN;
        zLower = bigExpFromLog(logZLowerPAC);
        zUpper = bigExpFromLog(logZUpperPAC);
        epsilon = epsilonFromLogBounds(logZLowerPAC, logZUpperPAC);
        setFrequencySeverityDiagnosticStatistics(
                finalCCD, selectedEta, finalLogR);
        certificateValid = finalInterval.isFinite()
                && isValidCertificate(zLower, zUpper, epsilon);
        certificateFailureReason = certificateValid ? ""
                : "AdaptiveFrequencySeverityPAC final interval failed validation";
        if (finalMeanExcessTest.rejected) {
            String reason = finalMeanExcessTest.logicalViolation
                    ? "AdaptiveFrequencySeverityPAC: final sample logically violates zero severity premise"
                    : "AdaptiveFrequencySeverityPAC: final severity p-value rejected premise "
                            + meanExcessPremiseId;
            reason += ", tail=" + finalMeanExcessTest.exceedanceCount
                    + "/" + finalInterval.sampleCount
                    + ", empiricalSeverity=" + finalInterval.empiricalConditionalMeanExcess
                    + ", logP=" + finalMeanExcessTest.logPValue
                    + ", alpha=" + meanExcessTestAlpha;

            failCertificate(reason, -1L);
        }
        if (!certificateValid) {
            printFinalSummary(startTime);
            return;
        }
        System.out.println("[PACK*-adaptive-frequency-severity] issued"
                + " assumption-conditional interval: premiseId="
                + meanExcessPremiseId
                + ", tail=" + finalInterval.exceedanceCount
                + "/" + finalInterval.sampleCount
                + ", epsilon="
                + String.format(Locale.ROOT, "%.9f", epsilon)
                + ", targetReached="
                + (epsilon <= targetEpsilon + 1.0e-12));
        printFinalSummary(startTime);
    }

    private double computeFrequencySeverityTrainingLogMu(
            List<CCDResult> samples, EtaCorrections eta) {
        double[] logA = new double[samples.size()];
        double[] logV = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            CCDResult sample = samples.get(i);
            double targetEnergy = sample.eMin
                    + computeEtaEnergy(sample, eta);
            logA[i] = meanExcessSourceLogWeight(
                    sample, targetEnergy);
            logV[i] = logA[i]
                    + (targetEnergy - sample.eTrue) / RT;
        }
        return logSumExp(logV) - logSumExp(logA);
    }

    private double[] computeFrequencySeverityLogRelativeWeights(
            List<CCDResult> samples, EtaCorrections eta,
            double logMuTrain) {
        double[] logRelative = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            CCDResult sample = samples.get(i);
            double etaEnergy = computeEtaEnergy(sample, eta);
            logRelative[i] = (sample.eMin + etaEnergy
                    - sample.eTrue) / RT - logMuTrain;
        }
        return logRelative;
    }

    private void setFrequencySeverityDiagnosticStatistics(
            List<CCDResult> samples, EtaCorrections eta,
            double[] logRelative) {
        double sumResidual = 0.0;
        double sumResidual2 = 0.0;
        for (CCDResult sample : samples) {
            double eEta = sample.eMin
                    + computeEtaEnergy(sample, eta);
            double residual = sample.eTrue - eEta;
            sumResidual += residual;
            sumResidual2 += residual * residual;
        }
        int n = samples.size();
        meanResidual = sumResidual / n;
        stdResidual = n > 1
                ? Math.sqrt(Math.max(0.0,
                (sumResidual2 - n * meanResidual * meanResidual)
                        / (n - 1))) : 0.0;
        double logMeanR = logSumExp(logRelative) - Math.log(n);
        double[] twice = new double[n];
        for (int i = 0; i < n; i++) twice[i] = 2.0 * logRelative[i];
        double logM2 = logSumExp(twice) - Math.log(n)
                - 2.0 * logMeanR;
        double m2 = logM2 < Math.log(Double.MAX_VALUE)
                ? Math.max(1.0, Math.exp(logM2))
                : Double.POSITIVE_INFINITY;
        cvPsi = Double.isFinite(m2)
                ? Math.sqrt(Math.max(0.0, m2 - 1.0))
                : Double.MAX_VALUE;
    }

    /** Keep exception details on one diagnostic log line. */
    private static String sanitizeTsv(String value) {
        if (value == null) return "";
        return value.replace('\t', ' ')
                .replace('\n', ' ')
                .replace('\r', ' ');
    }

    private boolean validateObservedLowerBound(List<CCDResult> samples, String stage) {
        int violations = 0;
        double minGap = Double.POSITIVE_INFINITY;
        for (CCDResult sample : samples) {
            if (!Double.isFinite(sample.eTrue) || !Double.isFinite(sample.eMin)) {
                System.out.println("[PACK*-lower-bound] ERROR: non-finite energy at " + stage);
                return false;
            }
            double gap = sample.eTrue - sample.eMin;
            minGap = Math.min(minGap, gap);
            if (gap < 0.0) violations++;
        }
        System.out.println("[PACK*-lower-bound] audit " + stage
                + ": samples=" + samples.size()
                + ", min(E_true-E_m)="
                + String.format(Locale.ROOT, "%.9f", minGap)
                + " kcal/mol, strictThreshold=0"
                + ", violations=" + violations);
        if (violations > 0) {
            System.out.println("[PACK*-lower-bound] ERROR: observed E_m > E_true. "
                    + "This falsifies the structural lower-bound premise for the configured "
                    + "Hamiltonian/interaction set; refusing certification.");
        }
        return violations == 0;
    }

    static double logAddExp(double first, double second) {
        if (first == Double.NEGATIVE_INFINITY) return second;
        if (second == Double.NEGATIVE_INFINITY) return first;
        if (!Double.isFinite(first) || !Double.isFinite(second)) {
            return first == Double.POSITIVE_INFINITY || second == Double.POSITIVE_INFINITY
                    ? Double.POSITIVE_INFINITY : Double.NaN;
        }
        double max = Math.max(first, second);
        return max + Math.log1p(Math.exp(Math.min(first, second) - max));
    }

    static boolean isValidCertificate(BigDecimal lower, BigDecimal upper,
                                      double epsilon) {
        return lower != null && upper != null
                && MathTools.isFinite(lower) && MathTools.isFinite(upper)
                && lower.signum() >= 0 && upper.signum() > 0
                && lower.compareTo(upper) <= 0
                && Double.isFinite(epsilon)
                && epsilon >= 0.0 && epsilon <= 1.0;
    }

    private void failCertificate(String reason, long startTime) {
        certificateValid = false;
        certificateFailureReason = reason;
        setZeroBounds(reason);
        if (startTime >= 0L) printFinalSummary(startTime);
    }

    /**
     * Compute the estimation sample size N* needed to reach targetEpsilon, by
     * inverting the empirical Bernstein bound on the residual-leg weight
     * w = exp(-xi/RT) using pilot estimates of its mean/variance and a
     * deterministic residual-weight range.
     *
     * epsilon = 2*Delta/(meanW + Delta)  =>  target Delta = eps*meanW/(2-eps).
     * Returns the smallest N in [2, maxEstSamples] whose Bernstein Delta <= target.
     * The pilot variance is inflated by nstarInflate for a conservative choice.
     */
    private List<int[]> sampleConformationsFromDP(int n, Random rng) {
        int numPos = rcs.getNumPos();
        int[][] confs = new int[n][numPos];
        for (int[] conf : confs) {
            Arrays.fill(conf, -1);
        }

        SplittableRandom[] sampleRngs = new SplittableRandom[n];
        long masterSeed = rng.nextLong();
        for (int s = 0; s < n; s++) {
            sampleRngs[s] = new SplittableRandom(mix64(masterSeed + 0x9E3779B97F4A7C15L * (long) s));
        }

        List<RootedTreeEdge> topDownOrder = getSamplingTopDownOrder();
        ExecutorService samplingPool = getSamplingPool();

        int edgeOrdinal = 0;
        for (RootedTreeEdge edge : topDownOrder) {
            if (!edge.getIsLambdaEdge()) continue;

            long edgeStart = System.currentTimeMillis();
            Map<Long, SampleGroup> groups = groupSamplesByMIdx(edge, confs);
            long edgeSeed = mix64(masterSeed + 0xD1B54A32D192ED03L * (long) edgeOrdinal++);
            sampleEdgeGroups(edge, confs, sampleRngs, groups, samplingPool, edgeSeed);
            logSamplingEdgeProgress(edge, n, groups, System.currentTimeMillis() - edgeStart);
        }

        List<int[]> conformations = new ArrayList<>(n);
        for (int[] conf : confs) {
            boolean valid = true;
            for (int p = 0; p < numPos; p++) {
                if (conf[p] < 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                conformations.add(conf);
            }
        }

        return conformations;
    }

    private Map<Long, SampleGroup> groupSamplesByMIdx(RootedTreeEdge edge, int[][] confs) {
        Map<Long, SampleGroup> groups = new HashMap<>();
        int[] mPositions = edge.getMPositionsSorted();
        int[] mLocalRCs = new int[mPositions.length];

        for (int s = 0; s < confs.length; s++) {
            for (int i = 0; i < mPositions.length; i++) {
                int pos = mPositions[i];
                int globalRC = confs[s][pos];
                if (globalRC < 0) {
                    throw new IllegalStateException("PACK* sampling reached edge before M position "
                            + pos + " was assigned");
                }
                mLocalRCs[i] = findLocalRCIndex(pos, globalRC);
            }
            long mIdx = edge.computeIndexInA(mLocalRCs);
            SampleGroup group = groups.get(mIdx);
            if (group == null) {
                group = new SampleGroup(mIdx);
                group.mRCs = Arrays.copyOf(mLocalRCs, mLocalRCs.length);
                groups.put(mIdx, group);
            }
            group.add(s);
        }

        return groups;
    }

    private void sampleEdgeGroups(RootedTreeEdge edge, int[][] confs,
                                  SplittableRandom[] sampleRngs,
                                  Map<Long, SampleGroup> groups,
                                  ExecutorService samplingPool,
                                  long edgeSeed) {
        int totalLambda = edge.getTotalLambdaStates();
        if (totalLambda <= 0) {
            throw new IllegalStateException("Lambda edge has no lambda states");
        }
        if (totalLambda == 1) {
            for (SampleGroup group : groups.values()) {
                for (int i = 0; i < group.count; i++) {
                    writeLambdaState(edge, 0, confs[group.sampleIndices[i]]);
                }
            }
            return;
        }

        if (trySampleEdgeGroupsGpu(edge, groups, confs, edgeSeed)) {
            return;
        }

        if (samplingPool == null || samplingThreads <= 1) {
            for (SampleGroup group : groups.values()) {
                sampleGroupSerial(edge, group, confs, sampleRngs);
            }
            return;
        }

        if (totalLambda >= samplingLargeLambdaThreshold) {
            processSingletonGroupsParallel(edge, groups.values(), confs, sampleRngs, samplingPool);
            for (SampleGroup group : groups.values()) {
                if (group.count >= 2) {
                    sampleGroupWithParallelCDF(edge, group, confs, sampleRngs, samplingPool);
                }
            }
        } else {
            processSmallGroupsParallel(edge, groups.values(), confs, sampleRngs, samplingPool);
        }
    }

    private boolean trySampleEdgeGroupsGpu(RootedTreeEdge edge,
                                           Map<Long, SampleGroup> groups,
                                           int[][] confs,
                                           long edgeSeed) {
        if (!gpuSampling || activeProposalEmat.hasHigherOrderTerms()
                || !edge.canUseGpuSampling()) {
            return false;
        }

        long[] mIdxPerSample = new long[confs.length];
        for (SampleGroup group : groups.values()) {
            for (int i = 0; i < group.count; i++) {
                mIdxPerSample[group.sampleIndices[i]] = group.mIdx;
            }
        }

        int[] lIdxPerSample = edge.sampleLambdaStatesGpu(
                mIdxPerSample, edgeSeed, samplingProgress, gpuSamplingRequestCache);
        if (lIdxPerSample == null || lIdxPerSample.length != confs.length) {
            return false;
        }

        for (int sampleIndex = 0; sampleIndex < confs.length; sampleIndex++) {
            writeLambdaState(edge, lIdxPerSample[sampleIndex], confs[sampleIndex]);
        }
        return true;
    }

    private void processSingletonGroupsParallel(RootedTreeEdge edge,
                                                Collection<SampleGroup> groups,
                                                int[][] confs,
                                                SplittableRandom[] sampleRngs,
                                                ExecutorService samplingPool) {
        List<SampleGroup> singletonGroups = new ArrayList<>();
        for (SampleGroup group : groups) {
            if (group.count == 1) singletonGroups.add(group);
        }
        runSamplingGroupsParallel(singletonGroups, group -> {
                int sampleIndex = group.sampleIndices[0];
                int lIdx = sampleLambdaStateStreaming(
                        edge, group.mIdx, group.mRCs,
                        sampleRngs[sampleIndex]);
                writeLambdaState(edge, lIdx, confs[sampleIndex]);
            }, samplingPool);
    }

    private void processSmallGroupsParallel(RootedTreeEdge edge,
                                            Collection<SampleGroup> groups,
                                            int[][] confs,
                                            SplittableRandom[] sampleRngs,
                                            ExecutorService samplingPool) {
        runSamplingGroupsParallel(groups,
                group -> sampleGroupSerial(edge, group, confs, sampleRngs),
                samplingPool);
    }

    /**
     * Submit at most one task per sampling worker.  The old implementation
     * submitted one future per M-group, which made a high-cardinality edge
     * build a large executor queue even though groups are independent.
     */
    private void runSamplingGroupsParallel(Collection<SampleGroup> groups,
        Consumer<SampleGroup> task,
                                           ExecutorService samplingPool) {
        if (groups.isEmpty()) return;
        List<SampleGroup> work = new ArrayList<>(groups);
        int workers = Math.min(samplingThreads, work.size());
        AtomicInteger next = new AtomicInteger(0);
        List<Future<?>> futures = new ArrayList<>(workers);
        for (int worker = 0; worker < workers; worker++) {
            futures.add(samplingPool.submit(() -> {
                int index;
                while ((index = next.getAndIncrement()) < work.size()) {
                    task.accept(work.get(index));
                }
            }));
        }
        waitForSamplingFutures(futures);
    }

    private void sampleGroupSerial(RootedTreeEdge edge, SampleGroup group,
                                   int[][] confs, SplittableRandom[] sampleRngs) {
        if (group.count == 1) {
            int sampleIndex = group.sampleIndices[0];
            int lIdx = sampleLambdaStateStreaming(
                    edge, group.mIdx, group.mRCs,
                    sampleRngs[sampleIndex]);
            writeLambdaState(edge, lIdx, confs[sampleIndex]);
            return;
        }

        double[] cdf = conditionalCDF(edge, group, null);
        sampleGroupFromCDF(edge, group, confs, sampleRngs, cdf);
    }

    private void sampleGroupWithParallelCDF(RootedTreeEdge edge, SampleGroup group,
                                            int[][] confs, SplittableRandom[] sampleRngs,
                                            ExecutorService samplingPool) {
        double[] cdf = conditionalCDF(edge, group, samplingPool);
        sampleGroupFromCDF(edge, group, confs, sampleRngs, cdf);
    }

    private record ConditionalCdfKey(RootedTreeEdge edge, long mIdx, boolean parallel) {}

    private double[] conditionalCDF(RootedTreeEdge edge, SampleGroup group,
                                    ExecutorService pool) {
        ConditionalCdfKey key = new ConditionalCdfKey(edge, group.mIdx, pool != null);
        double[] cached = conditionalCdfCache.get(key);
        if (cached != null) return cached;
        double[] cdf = pool == null
                ? buildConditionalCDFSerial(edge, group.mIdx, group.mRCs)
                : buildConditionalCDFParallel(edge, group.mIdx, group.mRCs, pool);
        long bytes = 160L + Double.BYTES * (long) cdf.length;
        // Builders use worker-local scratch. Only the retained copy is immutable.
        if (conditionalCdfCache.canStore(bytes)) {
            conditionalCdfCache.put(key, cdf.clone(), bytes);
        }
        return cdf;
    }

    private void invalidateSamplingCaches() {
        conditionalCdfCache.clear();
        gpuSamplingRequestCache.clear();
    }

    private void sampleGroupFromCDF(RootedTreeEdge edge, SampleGroup group,
                                    int[][] confs, SplittableRandom[] sampleRngs,
                                    double[] cdf) {
        double total = cdf[cdf.length - 1];
        if (!(total > 0.0) || !Double.isFinite(total)) {
            for (int i = 0; i < group.count; i++) {
                int sampleIndex = group.sampleIndices[i];
                int lIdx = sampleLambdaStateStreaming(
                        edge, group.mIdx, group.mRCs,
                        sampleRngs[sampleIndex]);
                writeLambdaState(edge, lIdx, confs[sampleIndex]);
            }
            return;
        }

        for (int i = 0; i < group.count; i++) {
            int sampleIndex = group.sampleIndices[i];
            int lIdx = sampleFromCDF(cdf, total, sampleRngs[sampleIndex]);
            writeLambdaState(edge, lIdx, confs[sampleIndex]);
        }
    }

    private double[] buildConditionalCDFSerial(
            RootedTreeEdge edge, long mIdx, int[] mRCs) {
        int totalLambda = edge.getTotalLambdaStates();
        double[] cdf = acquireSamplingCdf(totalLambda);

        double maxLog = Double.NEGATIVE_INFINITY;
        for (int lIdx = 0; lIdx < totalLambda; lIdx++) {
            double logWeight = computeLambdaLogWeight(edge, mIdx, mRCs, lIdx);
            cdf[lIdx] = logWeight;
            if (logWeight > maxLog) {
                maxLog = logWeight;
            }
        }

        if (!Double.isFinite(maxLog)) {
            return cdf;
        }

        double running = 0.0;
        for (int lIdx = 0; lIdx < totalLambda; lIdx++) {
            running += Math.exp(cdf[lIdx] - maxLog);
            cdf[lIdx] = running;
        }
        return cdf;
    }

    private double[] buildConditionalCDFParallel(
            RootedTreeEdge edge, long mIdx, int[] mRCs,
            ExecutorService samplingPool) {
        int totalLambda = edge.getTotalLambdaStates();
        double[] cdf = acquireSamplingCdf(totalLambda);
        int chunks = Math.min(samplingThreads, totalLambda);
        double[] chunkMax = new double[chunks];
        Arrays.fill(chunkMax, Double.NEGATIVE_INFINITY);

        runParallelChunks(samplingPool, chunks, chunk -> {
            int start = chunkStart(totalLambda, chunks, chunk);
            int end = chunkStart(totalLambda, chunks, chunk + 1);
            double max = Double.NEGATIVE_INFINITY;
            for (int lIdx = start; lIdx < end; lIdx++) {
                double logWeight = computeLambdaLogWeight(edge, mIdx, mRCs, lIdx);
                cdf[lIdx] = logWeight;
                if (logWeight > max) {
                    max = logWeight;
                }
            }
            chunkMax[chunk] = max;
        });

        double maxLog = Double.NEGATIVE_INFINITY;
        for (double max : chunkMax) {
            if (max > maxLog) {
                maxLog = max;
            }
        }
        if (!Double.isFinite(maxLog)) {
            return cdf;
        }

        final double globalMaxLog = maxLog;
        double[] chunkSums = new double[chunks];
        runParallelChunks(samplingPool, chunks, chunk -> {
            int start = chunkStart(totalLambda, chunks, chunk);
            int end = chunkStart(totalLambda, chunks, chunk + 1);
            double sum = 0.0;
            for (int lIdx = start; lIdx < end; lIdx++) {
                double weight = Math.exp(cdf[lIdx] - globalMaxLog);
                cdf[lIdx] = weight;
                sum += weight;
            }
            chunkSums[chunk] = sum;
        });

        double[] chunkOffsets = new double[chunks];
        double offset = 0.0;
        for (int chunk = 0; chunk < chunks; chunk++) {
            chunkOffsets[chunk] = offset;
            offset += chunkSums[chunk];
        }

        runParallelChunks(samplingPool, chunks, chunk -> {
            int start = chunkStart(totalLambda, chunks, chunk);
            int end = chunkStart(totalLambda, chunks, chunk + 1);
            double running = chunkOffsets[chunk];
            for (int lIdx = start; lIdx < end; lIdx++) {
                running += cdf[lIdx];
                cdf[lIdx] = running;
            }
        });

        return cdf;
    }

    private double[] acquireSamplingCdf(int length) {
        double[] cdf = samplingCdfScratch.get();
        // Callers use the array length as the lambda-state count, so retaining
        // a larger buffer across differently sized edges would be incorrect.
        if (cdf.length != length) {
            cdf = new double[length];
            samplingCdfScratch.set(cdf);
        }
        return cdf;
    }

    private int sampleLambdaStateStreaming(
            RootedTreeEdge edge, long mIdx, int[] mRCs,
            SplittableRandom rng) {
        int totalLambda = edge.getTotalLambdaStates();
        if (totalLambda <= 1) {
            return 0;
        }

        int bestIdx = totalLambda - 1;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int lIdx = 0; lIdx < totalLambda; lIdx++) {
            double logWeight = computeLambdaLogWeight(edge, mIdx, mRCs, lIdx);
            if (Double.isNaN(logWeight)) {
                continue;
            }
            double score = logWeight + sampleGumbel(rng);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = lIdx;
            }
        }
        return bestIdx;
    }

    private double computeLambdaLogWeight(RootedTreeEdge edge, long mIdx,
                                          int[] mRCs, int lIdx) {
        return edge.computeLambdaLogWeightPublic(
                mIdx, mRCs, lIdx, activeProposalEmat,
                interactionGraph, RT);
    }

    private void writeLambdaState(RootedTreeEdge edge, int lIdx, int[] conf) {
        edge.writeLambdaStateGlobal(lIdx, conf);
    }

    private int sampleFromCDF(double[] cdf, double total, SplittableRandom rng) {
        double target = rng.nextDouble() * total;
        int lo = 0;
        int hi = cdf.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (target < cdf[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private double sampleGumbel(SplittableRandom rng) {
        double u = rng.nextDouble();
        if (u <= 0.0) {
            u = Double.MIN_VALUE;
        }
        return -Math.log(-Math.log(u));
    }

    private int chunkStart(int total, int chunks, int chunk) {
        return (int) (((long) total * (long) chunk) / (long) chunks);
    }

    private void runParallelChunks(ExecutorService pool, int chunks, IntConsumer task) {
        List<Future<?>> futures = new ArrayList<>(chunks);
        for (int chunk = 0; chunk < chunks; chunk++) {
            final int c = chunk;
            futures.add(pool.submit(() -> task.accept(c)));
        }
        waitForSamplingFutures(futures);
    }

    private void waitForSamplingFutures(List<Future<?>> futures) {
        try {
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("PACK* sampling interrupted", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            throw new RuntimeException("PACK* sampling failed", cause);
        }
    }

    private void logSamplingEdgeProgress(RootedTreeEdge edge, int requestedSamples,
                                         Map<Long, SampleGroup> groups,
                                         long elapsedMs) {
        if (!samplingProgress) {
            return;
        }

        int totalLambda = edge.getTotalLambdaStates();
        if (totalLambda < samplingLargeLambdaThreshold && elapsedMs < 1000L) {
            return;
        }

        int singletonGroups = 0;
        int maxGroup = 0;
        for (SampleGroup group : groups.values()) {
            if (group.count == 1) {
                singletonGroups++;
            }
            maxGroup = Math.max(maxGroup, group.count);
        }

        System.out.println("[PACK*] Phase 1 edge: lambdaStates=" + totalLambda
                + ", samples=" + requestedSamples
                + ", distinctMIdx=" + groups.size()
                + ", singletonGroups=" + singletonGroups
                + ", maxGroup=" + maxGroup
                + ", threads=" + samplingThreads
                + ", elapsed=" + elapsedMs + " ms");
    }

    /**
     * Find the local RC index for a position given its global RC.
     */
    private int findLocalRCIndex(int pos, int globalRC) {
        int local = localRCByGlobalRC[pos].get(globalRC);
        if (local >= 0) {
            return local;
        }
        throw new IllegalStateException("Global RC " + globalRC + " not found at position " + pos);
    }

    /**
     * Build top-down traversal order for the compact tree.
     */
    private List<RootedTreeEdge> buildTopDownOrder(RootedTreeEdge rootEdge) {
        List<RootedTreeEdge> order = new ArrayList<>();
        Queue<RootedTreeEdge> queue = new ArrayDeque<>();
        queue.add(rootEdge);

        while (!queue.isEmpty()) {
            RootedTreeEdge edge = queue.poll();
            if (edge.getIsLambdaEdge()) {
                order.add(edge);
            }
            // Add compact children
            RootedTreeNode leftChild = edge.getCompactLeftChild();
            RootedTreeNode rightChild = edge.getCompactRightChild();
            if (leftChild != null && leftChild.getChildOfEdge() != null) {
                queue.add(leftChild.getChildOfEdge());
            }
            if (rightChild != null && rightChild.getChildOfEdge() != null) {
                queue.add(rightChild.getChildOfEdge());
            }
        }

        return order;
    }

    private List<RootedTreeEdge> getSamplingTopDownOrder() {
        if (samplingTopDownOrder == null || samplingOrderRoot != rootedRootEdge) {
            samplingTopDownOrder = List.copyOf(buildTopDownOrder(rootedRootEdge));
            samplingOrderRoot = rootedRootEdge;
        }
        return samplingTopDownOrder;
    }

    private ExecutorService getSamplingPool() {
        if (samplingThreads <= 1) return null;
        if (samplingResourcesClosed) {
            throw new IllegalStateException("PACK* sampling resources already closed");
        }
        if (samplingPool == null) {
            samplingPool = Executors.newFixedThreadPool(samplingThreads,
                    daemonThreadFactory("packstar-sample"));
        }
        return samplingPool;
    }

    private void shutdownSamplingResources() {
        if (samplingResourcesClosed) return;
        samplingResourcesClosed = true;
        if (samplingPool != null) {
            samplingPool.shutdownNow();
            samplingPool = null;
        }
        samplingCdfScratch.remove();
        System.out.println("[PACK*-perf] ccdCacheHits=" + totalCCDCacheHits
                + ", ccdCacheBytes=" + ccdEnergyCache.bytes()
                + ", cdfCacheHits=" + conditionalCdfCache.hits()
                + ", cdfCacheMisses=" + conditionalCdfCache.misses()
                + ", cdfCacheBytes=" + conditionalCdfCache.bytes()
                + ", gpuPreparedHits=" + gpuSamplingRequestCache.hits());
        ccdEnergyCache.clear();
        invalidateSamplingCaches();
    }

    // ========== Phase 2: Parallel CCD minimization ==========

    /**
     * Immutable key for an exact discrete conformation assignment.  Using a
     * value key rather than {@code int[]} identity is important here because
     * DP sampling allocates a fresh array for every draw.  The assignment is
     * copied once into the key, so the map remains safe even if a caller later
     * reuses or mutates its input array.
     */
    private static final class ConformationKey {
        private final int[] values;
        private final int hash;

        ConformationKey(int[] values) {
            if (values == null) {
                throw new IllegalArgumentException(
                        "CCD conformation cannot be null");
            }
            this.values = values.clone();
            this.hash = Arrays.hashCode(this.values);
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof ConformationKey
                    && Arrays.equals(values,
                    ((ConformationKey) other).values);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    /**
     * Unique CCD work plus the inverse map needed to scatter results back to
     * the original IID sample order.  The returned sample list deliberately
     * keeps duplicate references to one CCDResult: all later reductions still
     * visit every draw, while feature caches can also avoid recomputing the
     * same forcefield decomposition for a repeated assignment.
     */
    static final class UniqueConformationBatch {
        final List<int[]> uniqueConformations;
        final int[] uniqueIndexBySample;

        UniqueConformationBatch(List<int[]> uniqueConformations,
                                int[] uniqueIndexBySample) {
            this.uniqueConformations = uniqueConformations;
            this.uniqueIndexBySample = uniqueIndexBySample;
        }
    }

    static UniqueConformationBatch deduplicateConformations(
            List<int[]> conformations) {
        if (conformations == null) {
            throw new IllegalArgumentException(
                    "CCD conformation list cannot be null");
        }

        Map<ConformationKey, Integer> indexByKey =
                new LinkedHashMap<>(Math.max(16, conformations.size()));
        List<int[]> unique = new ArrayList<>(conformations.size());
        int[] uniqueIndexBySample = new int[conformations.size()];

        for (int sample = 0; sample < conformations.size(); sample++) {
            int[] conformation = conformations.get(sample);
            ConformationKey key = new ConformationKey(conformation);
            Integer uniqueIndex = indexByKey.get(key);
            if (uniqueIndex == null) {
                uniqueIndex = unique.size();
                indexByKey.put(key, uniqueIndex);
                unique.add(conformation.clone());
            }
            uniqueIndexBySample[sample] = uniqueIndex;
        }

        return new UniqueConformationBatch(
                List.copyOf(unique), uniqueIndexBySample);
    }

    private static class CCDResult {
        final int[] conf;
        final double eTrue;
        final double eMin;
        final EnergyCalculator.EnergiedParametricMolecule epmol;
        final double sourceProposalEnergy;
        final double sourceProposalLogZ;
        final String sourceProposalId;
        SampleFeatures features;
        String assignmentKey;

        CCDResult(int[] conf, double eTrue, double eMin,
                  EnergyCalculator.EnergiedParametricMolecule epmol,
                  double sourceProposalEnergy,
                  double sourceProposalLogZ,
                  String sourceProposalId) {
            this.conf = conf;
            this.eTrue = eTrue;
            this.eMin = eMin;
            this.epmol = epmol;
            this.sourceProposalEnergy = sourceProposalEnergy;
            this.sourceProposalLogZ = sourceProposalLogZ;
            this.sourceProposalId = sourceProposalId;
        }
    }

    /** Immutable target data only: proposal provenance and molecule objects are never cached. */
    private static final class CachedTargetEnergy {
        final double eTrue, eMin;
        final SampleFeatures features;
        CachedTargetEnergy(CCDResult result) {
            eTrue = result.eTrue;
            eMin = result.eMin;
            features = result.features;
        }
        long retainedBytes(int positions) {
            return 256L + 8L * positions + (features == null ? 0L
                    : 128L + 8L * features.oneBodyCorrections.length
                    + 8L * features.pairCorrections.length + 8L * features.pairKeys.length);
        }
    }

    /**
     * Eta-training features for one minimized conformation.  These are raw
     * forcefield-minus-base-EMAT term corrections and do not depend on the
     * fitted eta, shrinkage choice, or fold.  Keeping them on CCDResult makes
     * every later refit an aggregation pass instead of another forcefield
     * decomposition pass.
     */
    private static class SampleFeatures {
        final double[] oneBodyCorrections;
        final double[] pairCorrections;
        /** Pair-cell keys in the estimator's stable edge order. */
        final long[] pairKeys;

        SampleFeatures(double[] oneBodyCorrections,
                       double[] pairCorrections,
                       long[] pairKeys) {
            this.oneBodyCorrections = oneBodyCorrections;
            this.pairCorrections = pairCorrections;
            this.pairKeys = pairKeys;
        }
    }

    /**
     * Run CCD while retaining the exact proposal provenance of this batch.
     * The source energy and log normalizer are later used only by adaptation
     * to reweight samples to a selected proposal; final samples remain fresh.
     */
    private List<CCDResult> runParallelCCD(
            List<int[]> conformations,
            EnergyMatrix sourceProposalEmat,
            double sourceProposalLogZ,
            String sourceProposalId) {
        boolean recordSourceProposal = sourceProposalEmat != null
                && Double.isFinite(sourceProposalLogZ)
                && sourceProposalId != null
                && !sourceProposalId.isEmpty();
        UniqueConformationBatch uniqueBatch =
                deduplicateConformations(conformations);
        int total = conformations.size();
        int uniqueTotal = uniqueBatch.uniqueConformations.size();

        // Keep the estimator's external CCD count honest: it is the number of
        // actual minimizer invocations, not the number of IID records that
        // consume their results.  The latter is retained separately for audit
        // files and for checking the requested sample count.
        totalCCDSampleRecords += total;

        if (uniqueTotal < total) {
            System.out.println("[PACK*-CCD] exact conformation dedup: samples="
                    + total + ", unique=" + uniqueTotal
                    + ", reused=" + (total - uniqueTotal)
                    + ", saved=" + String.format(Locale.ROOT, "%.2f%%",
                    total > 0 ? 100.0 * (total - uniqueTotal) / total : 0.0));
        }

        AtomicReferenceArray<CCDResult> results =
                new AtomicReferenceArray<>(uniqueTotal);
        List<ConformationKey> keys = new ArrayList<>(uniqueTotal);
        int cacheHits = 0;
        for (int i = 0; i < uniqueTotal; i++) {
            int[] conf = uniqueBatch.uniqueConformations.get(i);
            ConformationKey key = new ConformationKey(conf);
            keys.add(key);
            CachedTargetEnergy cached = ccdEnergyCache.get(key);
            if (cached != null && (!recordSourceProposal || cached.features != null)) {
                CCDResult result = new CCDResult(conf, cached.eTrue, cached.eMin, null,
                        recordSourceProposal ? computeFullConfProposalEnergy(conf, sourceProposalEmat) : Double.NaN,
                        recordSourceProposal ? sourceProposalLogZ : Double.NaN,
                        recordSourceProposal ? sourceProposalId : null);
                result.features = cached.features;
                results.set(i, result);
                cacheHits++;
            }
        }
        int ccdCalls = uniqueTotal - cacheHits;
        totalCCDCalls += ccdCalls;
        totalCCDCacheHits += cacheHits;
        if (cacheHits > 0) {
            System.out.println("[PACK*-CCD] target cache: unique=" + uniqueTotal
                    + ", hits=" + cacheHits + ", newCCD=" + ccdCalls);
        }
        AtomicInteger completed = new AtomicInteger(0);

        for (int idx = 0; idx < uniqueTotal; idx++) {
            if (results.get(idx) != null) continue;
            minimizingEcalc.tasks.waitForCapacity(ccdSubmissionBatchSize);
            int[] conf = uniqueBatch.uniqueConformations.get(idx);
            final int ci = idx;
            ResidueInteractions inters = makeSparseFullConfInters(conf);
            RCTuple tuple = new RCTuple(conf);

            minimizingEcalc.calcEnergyAsync(tuple, inters, epmol -> {

                double eTrue = epmol.energy;
                double eMin = computeFullConfPairwiseEnergy(conf, branchMinimizingEmat);
                double sourceEnergy = recordSourceProposal
                        ? computeFullConfProposalEnergy(
                        conf, sourceProposalEmat)
                        : Double.NaN;
                // Callback completion order depends on task scheduling and GPU
                // load.  Preserve the seeded sample order so eta accumulation,
                // pilot statistics, and all later floating-point reductions do
                // not inherit that nondeterminism.
                CCDResult result = new CCDResult(
                        conf, eTrue, eMin, epmol,
                        sourceEnergy,
                        recordSourceProposal
                                ? sourceProposalLogZ : Double.NaN,
                        recordSourceProposal ? sourceProposalId : null);
                boolean validTarget = Double.isFinite(eTrue) && Double.isFinite(eMin)
                        && eTrue >= eMin;
                if (validTarget && recordSourceProposal) {
                    result.features = buildSampleFeatures(result);
                }
                if (validTarget) {
                    // Violating/nonfinite samples keep their pose for forensic diagnostics.
                    CCDResult compact = new CCDResult(conf, eTrue, eMin, null,
                            result.sourceProposalEnergy, result.sourceProposalLogZ, result.sourceProposalId);
                    compact.features = result.features;
                    result = compact;
                }
                results.set(ci, result);

                int done = completed.incrementAndGet();
                if (done % 100 == 0 || done == ccdCalls) {
                    System.out.println("[PACK*] CCD progress: " + done + "/"
                            + ccdCalls + " new unique (logical samples="
                            + total + ")");
                }
            });

        }

        minimizingEcalc.tasks.waitForFinish();

        // Commit in assignment order, so scheduling never changes LRU admission.
        for (int i = 0; i < uniqueTotal; i++) {
            CCDResult result = results.get(i);
            if (result != null && Double.isFinite(result.eTrue)
                    && Double.isFinite(result.eMin) && result.eTrue >= result.eMin) {
                CachedTargetEnergy cached = new CachedTargetEnergy(result);
                ccdEnergyCache.put(keys.get(i), cached, cached.retainedBytes(result.conf.length));
            }
        }
        List<CCDResult> orderedResults = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            CCDResult result = results.get(uniqueBatch.uniqueIndexBySample[i]);
            if (result == null) {
                throw new IllegalStateException(
                        "CCD result missing for unique conformation "
                                + uniqueBatch.uniqueIndexBySample[i]
                                + " (sample " + i + ")");
            }
            orderedResults.add(result);
        }
        return orderedResults;
    }

    private SampleFeatures getSampleFeatures(CCDResult result) {
        if (result.features == null) {
            result.features = buildSampleFeatures(result);
        }
        return result.features;
    }

    private static String getAssignmentKey(CCDResult result) {
        if (result.assignmentKey == null) {
            result.assignmentKey = Arrays.toString(result.conf);
        }
        return result.assignmentKey;
    }

    private SampleFeatures buildSampleFeatures(CCDResult result) {
        int[] conf = result.conf;
        // The general-DOF CUDA backend evaluates energies on the GPU, but
        // proposal features need the CPU forcefield's residue-subset view.
        // Reuse exactly the same parameters, interactions, and minimized pose.
        ResidueForcefieldEnergy efunc =
                minimizingEcalc.ecalc.type == EnergyCalculator.Type.ResidueCuda
                        ? new ResidueForcefieldEnergy(minimizingEcalc.ecalc.resPairCache,
                                result.epmol.inters, result.epmol.pmol.mol)
                        : (ResidueForcefieldEnergy) minimizingEcalc.ecalc
                                .makeEnergyFunction(result.epmol);
        double[] oneBody = new double[conf.length];
        Arrays.fill(oneBody, Double.NaN);
        for (int pos = 0; pos < conf.length; pos++) {
            int rc = conf[pos];
            if (rc < 0) continue;
            oneBody[pos] = efunc.makeSubset(
                    getSingleInteractionTemplate(pos, rc)).getEnergy()
                    - branchMinimizingEmat.getOneBody(pos, rc);
        }

        double[] pair = new double[interactionEdges.length];
        long[] pairKeys = new long[interactionEdges.length];
        Arrays.fill(pairKeys, MISSING_PAIR_KEY);
        Arrays.fill(pair, Double.NaN);
        for (int edgeIndex = 0; edgeIndex < interactionEdges.length;
             edgeIndex++) {
            int pos1 = interactionEdges[edgeIndex][0];
            int pos2 = interactionEdges[edgeIndex][1];
            int rc1 = conf[pos1];
            int rc2 = conf[pos2];
            if (rc1 < 0 || rc2 < 0) continue;
            pairKeys[edgeIndex] = EtaCorrections.packPairKey(
                    pos1, rc1, pos2, rc2);
            pair[edgeIndex] = efunc.makeSubset(
                    getPairInteractionTemplate(pos1, rc1, pos2, rc2))
                    .getEnergy()
                    - branchMinimizingEmat.getPairwise(
                    pos1, rc1, pos2, rc2);
        }
        return new SampleFeatures(oneBody, pair, pairKeys);
    }

    private double computeFullConfProposalEnergy(
            int[] conf, EnergyMatrix proposalEmat) {
        if (!proposalEmat.hasHigherOrderTerms()) {
            return computeFullConfPairwiseEnergy(conf, proposalEmat);
        }
        return proposalEmat.getInternalEnergy(new RCTuple(conf));
    }

    // ========== Phase 3: Extract per-term eta corrections ==========

    private static class EtaCorrections {
        @FunctionalInterface
        interface PairStatsConsumer {
            void accept(long key, double sum, int count);
        }

        /** Primitive long-to-int map for distinct-context coverage counts. */
        static class PairCountMap {
            private static final double LOAD_FACTOR = 0.70;
            private long[] keys;
            private int[] values;
            private boolean[] used;
            private int mask;
            private int threshold;
            private int size;

            PairCountMap() {
                allocate(16);
            }

            void increment(long key) {
                int slot = findSlot(key);
                if (!used[slot]) {
                    ensureCapacityForInsert();
                    slot = findSlot(key);
                    used[slot] = true;
                    keys[slot] = key;
                    size++;
                }
                if (values[slot] == Integer.MAX_VALUE) {
                    throw new ArithmeticException(
                            "eta context count overflow");
                }
                values[slot]++;
            }

            int getOrDefault(long key, int defaultValue) {
                int slot = findSlot(key);
                return used[slot] ? values[slot] : defaultValue;
            }

            private void allocate(int capacity) {
                keys = new long[capacity];
                values = new int[capacity];
                used = new boolean[capacity];
                mask = capacity - 1;
                threshold = Math.max(1,
                        (int) Math.floor(capacity * LOAD_FACTOR));
                size = 0;
            }

            private void ensureCapacityForInsert() {
                if (size + 1 <= threshold) return;
                long[] oldKeys = keys;
                int[] oldValues = values;
                boolean[] oldUsed = used;
                if (oldKeys.length > (1 << 29)) {
                    throw new IllegalStateException(
                            "eta context table exceeds supported capacity");
                }
                allocate(oldKeys.length << 1);
                for (int oldSlot = 0; oldSlot < oldUsed.length; oldSlot++) {
                    if (!oldUsed[oldSlot]) continue;
                    int slot = findSlot(oldKeys[oldSlot]);
                    used[slot] = true;
                    keys[slot] = oldKeys[oldSlot];
                    values[slot] = oldValues[oldSlot];
                    size++;
                }
            }

            private int findSlot(long key) {
                int slot = PairStatsMap.mix(key) & mask;
                while (used[slot] && keys[slot] != key) {
                    slot = (slot + 1) & mask;
                }
                return slot;
            }
        }

        /**
         * Small open-addressed map for the sparse pair cells.  Pair eta is
         * updated in a single estimator thread, so a primitive table is enough
         * and avoids both boxed Long keys and one double[2] allocation per
         * observed cell.  Iteration order is intentionally unspecified; all
         * serialized views sort their keys when order matters.
         */
        static class PairStatsMap {
            private static final double LOAD_FACTOR = 0.70;
            private long[] keys;
            private double[] sums;
            private int[] counts;
            private boolean[] used;
            private int mask;
            private int threshold;
            private int size;

            PairStatsMap() {
                this(16);
            }

            PairStatsMap(int requestedCapacity) {
                int capacity = 1;
                while (capacity < Math.max(4, requestedCapacity)) {
                    capacity <<= 1;
                }
                allocate(capacity);
            }

            int size() {
                return size;
            }

            void add(long key, double value) {
                int slot = findSlot(key);
                if (!used[slot]) {
                    ensureCapacityForInsert();
                    slot = findSlot(key);
                    used[slot] = true;
                    keys[slot] = key;
                    size++;
                }
                sums[slot] += value;
                if (counts[slot] == Integer.MAX_VALUE) {
                    throw new ArithmeticException(
                            "eta pair observation count overflow");
                }
                counts[slot]++;
            }

            void put(long key, double sum, int count) {
                if (count < 0) {
                    throw new IllegalArgumentException(
                            "eta pair observation count must be nonnegative");
                }
                int slot = findSlot(key);
                if (!used[slot]) {
                    ensureCapacityForInsert();
                    slot = findSlot(key);
                    used[slot] = true;
                    keys[slot] = key;
                    size++;
                }
                sums[slot] = sum;
                counts[slot] = count;
            }

            int count(long key) {
                int slot = findSlot(key);
                return used[slot] ? counts[slot] : 0;
            }

            double sum(long key) {
                int slot = findSlot(key);
                return used[slot] ? sums[slot] : 0.0;
            }

            double mean(long key) {
                int slot = findSlot(key);
                if (!used[slot] || counts[slot] == 0) return 0.0;
                return sums[slot] / counts[slot];
            }

            long[] keys() {
                long[] result = new long[size];
                int out = 0;
                for (int slot = 0; slot < used.length; slot++) {
                    if (used[slot]) result[out++] = keys[slot];
                }
                return result;
            }

            void forEach(PairStatsConsumer consumer) {
                for (int slot = 0; slot < used.length; slot++) {
                    if (used[slot]) {
                        consumer.accept(keys[slot], sums[slot], counts[slot]);
                    }
                }
            }

            private void allocate(int capacity) {
                keys = new long[capacity];
                sums = new double[capacity];
                counts = new int[capacity];
                used = new boolean[capacity];
                mask = capacity - 1;
                threshold = Math.max(1,
                        (int) Math.floor(capacity * LOAD_FACTOR));
                size = 0;
            }

            private void ensureCapacityForInsert() {
                if (size + 1 <= threshold) return;
                long[] oldKeys = keys;
                double[] oldSums = sums;
                int[] oldCounts = counts;
                boolean[] oldUsed = used;
                if (oldKeys.length > (1 << 29)) {
                    throw new IllegalStateException(
                            "eta pair table exceeds supported capacity");
                }
                allocate(oldKeys.length << 1);
                for (int oldSlot = 0; oldSlot < oldUsed.length; oldSlot++) {
                    if (!oldUsed[oldSlot]) continue;
                    int slot = findSlot(oldKeys[oldSlot]);
                    used[slot] = true;
                    keys[slot] = oldKeys[oldSlot];
                    sums[slot] = oldSums[oldSlot];
                    counts[slot] = oldCounts[oldSlot];
                    size++;
                }
            }

            private int findSlot(long key) {
                int slot = mix(key) & mask;
                while (used[slot] && keys[slot] != key) {
                    slot = (slot + 1) & mask;
                }
                return slot;
            }

            private static int mix(long value) {
                value ^= value >>> 33;
                value *= 0xff51afd7ed558ccdl;
                value ^= value >>> 33;
                value *= 0xc4ceb9fe1a85ec53l;
                value ^= value >>> 33;
                return (int) value;
            }
        }

        // eta_oneBody[pos][rc] = mean correction for one-body term
        final double[][] oneBody;
        final int[][] oneBodyCounts; // per-(pos,rc) counts, NOT per-pos
        // eta_pair[pos1][rc1][pos2][rc2] = mean correction for pair term
        // Stored as map to avoid sparse array issues
        final PairStatsMap pairSums;
        int oneBodyCount = 0;
        // Added once per complete conformation.  This is an energy gauge: it
        // changes Z_eta and residual weights by reciprocal constants but leaves
        // the normalized proposal q_eta exactly unchanged.
        double globalOffsetKcal = 0.0;

        EtaCorrections(int numPos, int[] numRCsPerPos) {
            this(numPos, numRCsPerPos, 16);
        }

        EtaCorrections(int numPos, int[] numRCsPerPos,
                       int expectedPairCells) {
            oneBody = new double[numPos][];
            oneBodyCounts = new int[numPos][];
            for (int p = 0; p < numPos; p++) {
                oneBody[p] = new double[numRCsPerPos[p]];
                oneBodyCounts[p] = new int[numRCsPerPos[p]];
            }
            pairSums = new PairStatsMap(expectedPairCells);
        }

        static long packPairKey(int pos1, int rc1, int pos2, int rc2) {
            // Ensure pos1 < pos2
            if (pos1 > pos2) {
                int tmp = pos1; pos1 = pos2; pos2 = tmp;
                tmp = rc1; rc1 = rc2; rc2 = tmp;
            }
            return ((long) pos1 << 48) | ((long) rc1 << 32) | ((long) pos2 << 16) | rc2;
        }

        static int pairPos1(long key) {
            return (int) ((key >>> 48) & 0xffffL);
        }

        static int pairRc1(long key) {
            return (int) ((key >>> 32) & 0xffffL);
        }

        static int pairPos2(long key) {
            return (int) ((key >>> 16) & 0xffffL);
        }

        static int pairRc2(long key) {
            return (int) (key & 0xffffL);
        }

        void addOneBodySample(int pos, int rc, double correction) {
            oneBody[pos][rc] += correction;
            oneBodyCounts[pos][rc]++;
        }

        void addPairSample(int pos1, int rc1, int pos2, int rc2, double correction) {
            long key = packPairKey(pos1, rc1, pos2, rc2);
            pairSums.add(key, correction);
        }

        double getOneBodyEta(int pos, int rc) {
            if (oneBodyCounts[pos][rc] == 0) return 0.0;
            return oneBody[pos][rc] / oneBodyCounts[pos][rc];
        }

        double getPairEta(long key) {
            return pairSums.mean(key);
        }

    }

    /**
     * Online two-fold fit.  Historical adaptation samples are fixed before the
     * newest on-policy batch exists, so both fold fits may use all history.
     * Each newest-batch observation is used by exactly one fold fit and scored
     * by the complementary fit.  This prevents a failed q_eta batch from being
     * evaluated as if it were an IID extension of the original q_m batch.
     */
    private MeanExcessEtaTraining extractFrequencySeverityEtaTraining(
            List<CCDResult> history,
            List<CCDResult> newestOnPolicyBatch) {
        int numPos = rcs.getNumPos();
        int[] numRCs = new int[numPos];
        for (int pos = 0; pos < numPos; pos++) {
            numRCs[pos] = branchMinimizingEmat.getNumConfAtPos(pos);
        }
        EtaCorrections all = new EtaCorrections(numPos, numRCs);
        EtaCorrections[] folds = new EtaCorrections[MEAN_EXCESS_FOLDS];
        MeanExcessEtaCoverage[] foldCoverage = new MeanExcessEtaCoverage[MEAN_EXCESS_FOLDS];
        Set<String>[] foldAssignments = new Set[MEAN_EXCESS_FOLDS];
        List<CCDResult>[] foldFitSamples = new List[MEAN_EXCESS_FOLDS];
        for (int fold = 0; fold < MEAN_EXCESS_FOLDS; fold++) {
            folds[fold] = new EtaCorrections(numPos, numRCs);
            foldCoverage[fold] = new MeanExcessEtaCoverage(numPos, numRCs);
            foldAssignments[fold] = new HashSet<>();
            foldFitSamples[fold] = new ArrayList<>();
        }
        List<CCDResult> allFitSamples = new ArrayList<>(
                history.size() + newestOnPolicyBatch.size());
        MeanExcessEtaCoverage allCoverage = new MeanExcessEtaCoverage(numPos, numRCs);
        Set<String> allAssignments = new HashSet<>();
        int totalSamples = history.size() + newestOnPolicyBatch.size();

        for (int sampleIndex = 0; sampleIndex < totalSamples; sampleIndex++) {
            boolean historical = sampleIndex < history.size();
            int newestIndex = sampleIndex - history.size();
            CCDResult result = historical
                    ? history.get(sampleIndex)
                    : newestOnPolicyBatch.get(newestIndex);
            int newestFold = historical ? -1
                    : newestIndex % MEAN_EXCESS_FOLDS;
            allFitSamples.add(result);
            for (int fold = 0; fold < MEAN_EXCESS_FOLDS; fold++) {
                if (historical || fold == newestFold) {
                    foldFitSamples[fold].add(result);
                }
            }
            int[] conf = result.conf;
            {
            SampleFeatures features = getSampleFeatures(result);

            for (int pos = 0; pos < numPos; pos++) {
                int rc = conf[pos];
                if (rc < 0) continue;
                double correction = features.oneBodyCorrections[pos];
                all.addOneBodySample(pos, rc, correction);
                for (int fold = 0; fold < MEAN_EXCESS_FOLDS; fold++) {
                    if (historical || fold == newestFold) {
                        folds[fold].addOneBodySample(pos, rc, correction);
                    }
                }
            }
            for (int edgeIndex = 0; edgeIndex < interactionEdges.length;
                 edgeIndex++) {
                int pos1 = interactionEdges[edgeIndex][0];
                int pos2 = interactionEdges[edgeIndex][1];
                int rc1 = conf[pos1];
                int rc2 = conf[pos2];
                if (rc1 < 0 || rc2 < 0) continue;
                double correction = features.pairCorrections[edgeIndex];
                all.addPairSample(pos1, rc1, pos2, rc2, correction);
                for (int fold = 0; fold < MEAN_EXCESS_FOLDS; fold++) {
                    if (historical || fold == newestFold) {
                        folds[fold].addPairSample(
                                pos1, rc1, pos2, rc2, correction);
                    }
                }
            }
            }

            String assignment = getAssignmentKey(result);
            if (allAssignments.add(assignment)) {
                addFrequencySeverityCoverage(conf, allCoverage);
            }
            for (int fold = 0; fold < MEAN_EXCESS_FOLDS; fold++) {
                if ((historical || fold == newestFold)
                        && foldAssignments[fold].add(assignment)) {
                    addFrequencySeverityCoverage(conf, foldCoverage[fold]);
                }
            }
        }
        finalizeEtaTermCounts(all);
        for (EtaCorrections fold : folds) finalizeEtaTermCounts(fold);
        return new MeanExcessEtaTraining(
                all, folds, allCoverage, foldCoverage,
                List.copyOf(allFitSamples), foldFitSamples);
    }

    private void addFrequencySeverityCoverage(int[] conf, MeanExcessEtaCoverage coverage) {
        for (int pos = 0; pos < conf.length; pos++) {
            int rc = conf[pos];
            if (rc >= 0) coverage.unaryDistinctContexts[pos][rc]++;
        }
        for (int[] edge : interactionEdges) {
            int pos1 = edge[0];
            int pos2 = edge[1];
            int rc1 = conf[pos1];
            int rc2 = conf[pos2];
            if (rc1 < 0 || rc2 < 0) continue;
            long key = EtaCorrections.packPairKey(pos1, rc1, pos2, rc2);
            coverage.pairDistinctContexts.increment(key);
        }
    }

    private void finalizeEtaTermCounts(EtaCorrections eta) {
        eta.oneBodyCount = 0;
        for (int pos = 0; pos < eta.oneBody.length; pos++) {
            for (int rc = 0; rc < eta.oneBody[pos].length; rc++) {
                if (eta.oneBodyCounts[pos][rc] > 0) eta.oneBodyCount++;
            }
        }
    }

    /**
     * Extract per-term corrections from CCD results.
     *
     * For each minimized conformation, evaluate individual energy terms at the
     * CCD-optimized coordinates and compare with emat_min entries.
     *
     * Uses the forcefield breakdown approach from ConfAnalyzer.
     */
    // ========== Phase 4: Build corrected emat and recompute DP ==========

    /** Build E_corrected = E_m + eta on the fixed interaction graph. */
    private EnergyMatrix buildCorrectedEmat(EtaCorrections eta) {
        EnergyMatrix corrected = beginPairwiseCorrectedEmat();

        // A constant energy gauge must reach every conformation.
        if (eta.globalOffsetKcal != 0.0) {
            int numRC = corrected.getNumConfAtPos(0);
            for (int rc = 0; rc < numRC; rc++) {
                corrected.setOneBody(0, rc,
                        corrected.getOneBody(0, rc) + eta.globalOffsetKcal);
            }
        }

        // Record the sparse cells to restore before the next refit.
        ensureCorrectedOneBodyKeyCapacity(eta.oneBodyCount);
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            for (int rc = 0; rc < eta.oneBody[pos].length; rc++) {
                if (eta.oneBodyCounts[pos][rc] == 0) continue;
                correctedPairwiseOneBodyKeys[correctedPairwiseOneBodyCount++] =
                        ((long) pos << 32) | (rc & 0xffffffffL);
                corrected.setOneBody(pos, rc,
                        corrected.getOneBody(pos, rc) + eta.getOneBodyEta(pos, rc));
            }
        }

        ensureCorrectedPairKeyCapacity(eta.pairSums.size());
        eta.pairSums.forEach((key, sum, count) -> {
            if (count <= 0) return;
            int pos1 = EtaCorrections.pairPos1(key), rc1 = EtaCorrections.pairRc1(key);
            int pos2 = EtaCorrections.pairPos2(key), rc2 = EtaCorrections.pairRc2(key);
            correctedPairwisePairKeys[correctedPairwisePairCount++] = key;
            corrected.setPairwise(pos1, rc1, pos2, rc2,
                    corrected.getPairwise(pos1, rc1, pos2, rc2) + sum / count);
        });
        correctedPairwiseGlobalOffsetKcal = eta.globalOffsetKcal;
        return corrected;
    }

    /**
     * Start a pair-only refit on the reusable corrected matrix.  Restoring
     * touched cells from the immutable base matrix avoids round-off drift from
     * subtracting the previous correction and keeps the refit numerically
     * equivalent to a fresh {@code new EnergyMatrix(branchMinimizingEmat)}.
     */
    private EnergyMatrix beginPairwiseCorrectedEmat() {
        if (correctedPairwiseEmat == null) {
            correctedPairwiseEmat = new EnergyMatrix(branchMinimizingEmat);
            correctedPairwiseOneBodyCount = 0;
            correctedPairwisePairCount = 0;
            correctedPairwiseGlobalOffsetKcal = 0.0;
            return correctedPairwiseEmat;
        }

        if (correctedPairwiseGlobalOffsetKcal != 0.0) {
            int numRC = correctedPairwiseEmat.getNumConfAtPos(0);
            for (int rc = 0; rc < numRC; rc++) {
                correctedPairwiseEmat.setOneBody(
                        0, rc, branchMinimizingEmat.getOneBody(0, rc));
            }
        }
        for (int i = 0; i < correctedPairwiseOneBodyCount; i++) {
            long key = correctedPairwiseOneBodyKeys[i];
            int pos = (int) (key >>> 32);
            int rc = (int) key;
            correctedPairwiseEmat.setOneBody(
                    pos, rc, branchMinimizingEmat.getOneBody(pos, rc));
        }
        for (int i = 0; i < correctedPairwisePairCount; i++) {
            long key = correctedPairwisePairKeys[i];
            int pos1 = EtaCorrections.pairPos1(key);
            int rc1 = EtaCorrections.pairRc1(key);
            int pos2 = EtaCorrections.pairPos2(key);
            int rc2 = EtaCorrections.pairRc2(key);
            correctedPairwiseEmat.setPairwise(
                    pos1, rc1, pos2, rc2,
                    branchMinimizingEmat.getPairwise(
                            pos1, rc1, pos2, rc2));
        }
        correctedPairwiseOneBodyCount = 0;
        correctedPairwisePairCount = 0;
        correctedPairwiseGlobalOffsetKcal = 0.0;
        return correctedPairwiseEmat;
    }

    private void ensureCorrectedOneBodyKeyCapacity(int capacity) {
        if (correctedPairwiseOneBodyKeys.length < capacity) {
            correctedPairwiseOneBodyKeys = Arrays.copyOf(
                    correctedPairwiseOneBodyKeys,
                    Math.max(capacity,
                            Math.max(4, correctedPairwiseOneBodyKeys.length * 2)));
        }
    }

    private void ensureCorrectedPairKeyCapacity(int capacity) {
        if (correctedPairwisePairKeys.length < capacity) {
            correctedPairwisePairKeys = Arrays.copyOf(
                    correctedPairwisePairKeys,
                    Math.max(capacity,
                            Math.max(4, correctedPairwisePairKeys.length * 2)));
        }
    }

    private CorrectedDPResult recomputeDP(EnergyMatrix correctedEmat) {
        // Corrected matrices are mutated in place: object identity is not a version.
        invalidateSamplingCaches();
        // Keep the rigid lower-bound matrix unchanged and recompute the
        // corrected proposal tables directly. The only retained route never
        // reuses corrected DP tables across independently learned proposals.
        RootedTreeEdge.postOrderInitIncremental(
                rootedRoot, branchRigidEmat, correctedEmat,
                interactionGraph, RT);
        RootedTreeEdge.postOrderComputeFullDP(rootedRoot);
        activeProposalEmat = correctedEmat;
        return new CorrectedDPResult(rootedRootEdge.getLogZUpper(0));
    }

    public static double epsilonFromLogBounds(double logZLower, double logZUpper) {
        if (Double.isFinite(logZLower) && Double.isFinite(logZUpper)) {
            double orderingError = logZLower - logZUpper;
            if (orderingError > 1.0e-12) return 1.0;
            double eps = 1.0 - Math.exp(logZLower - logZUpper);
            if (eps < 0.0 && eps > -1e-12) eps = 0.0;
            if (eps > 1.0 && eps < 1.0 + 1e-12) eps = 1.0;
            return Double.isFinite(eps) && eps >= 0.0 && eps <= 1.0
                    ? eps : 1.0;
        }
        return 1.0;
    }

    private static String formatLog10(double logZ) {
        if (logZ == Double.NEGATIVE_INFINITY) return "-inf";
        if (logZ == Double.POSITIVE_INFINITY) return "+inf";
        if (Double.isNaN(logZ)) return "NaN";
        return String.format(Locale.ROOT, "%.6f", logZ / Math.log(10.0));
    }

    /**
     * Compute full conformation pairwise energy from an energy matrix.
     * Only sums over pairs in the interaction graph (sparse mode).
     */
    private double computeFullConfPairwiseEnergy(int[] conf, EnergyMatrix emat) {
        double energy = emat.getConstTerm();
        for (int i = 0; i < conf.length; i++) {
            if (conf[i] < 0) continue;
            energy += emat.getOneBody(i, conf[i]);
            for (int edgeIndex : interactionEdgeIndicesByFirstPosition[i]) {
                int pos2 = interactionEdges[edgeIndex][1];
                if (conf[pos2] < 0) continue;
                energy += emat.getPairwise(
                        i, conf[i], pos2, conf[pos2]);
            }
        }
        return energy;
    }

    /**
     * Build sparse residue interactions for a full conformation.
     */
    private ResidueInteractions makeSparseFullConfInters(int[] conf) {
        ResidueInteractions inters = new ResidueInteractions();
        for (int pos = 0; pos < conf.length; pos++) {
            if (conf[pos] < 0) continue;
            inters.addAll(getSingleInteractionTemplate(pos, conf[pos]));
        }
        for (int[] edge : interactionEdges) {
            int pos1 = edge[0];
            int pos2 = edge[1];
            if (conf[pos1] < 0 || conf[pos2] < 0) continue;
            inters.addAll(getPairInteractionTemplate(
                    pos1, conf[pos1], pos2, conf[pos2]));
        }
        if (minimizingEcalc.addShellInters) {
            inters.addAll(getShellInteractionTemplate());
        }
        return inters;
    }

    /**
     * Convert a log-space value to BigDecimal.
     */
    static BigDecimal bigExpFromLog(double logVal) {
        if (logVal == Double.NEGATIVE_INFINITY) return BigDecimal.ZERO;
        if (!Double.isFinite(logVal)) return MathTools.BigNaN;

        // Use the arbitrary-precision exponential path for very negative logs;
        // ordinary double-space exponentiation would underflow there.
        return LOG_SPACE_EXP.exp(logVal);
    }

    private void setZeroBounds(String reason) {
        System.out.println("[PACK*] WARNING: " + reason + "; returning zero Z bounds");
        certificateValid = false;
        certificateFailureReason = reason;
        zLower = BigDecimal.ZERO;
        zUpper = BigDecimal.ZERO;
        logZLowerPAC = Double.NEGATIVE_INFINITY;
        logZUpperPAC = Double.NEGATIVE_INFINITY;
        epsilon = 1.0;
        cvPsi = Double.MAX_VALUE;
        meanResidual = 0.0;
        stdResidual = 0.0;
    }

    // ========== Result getters ==========

    public BigDecimal getZLower() { return zLower; }
    public BigDecimal getZUpper() { return zUpper; }
    public int getTotalCCDCalls() { return totalCCDCalls; }
    /** Number of logical CCD sample records, including deduplicated repeats. */
    public int getTotalCCDSampleRecords() { return totalCCDSampleRecords; }
    public double getCvPsi() { return cvPsi; }
    public double getMeanResidual() { return meanResidual; }
    public double getStdResidual() { return stdResidual; }
    public boolean hasValidCertificate() { return certificateValid; }
    public String getCertificateFailureReason() { return certificateFailureReason; }
}
