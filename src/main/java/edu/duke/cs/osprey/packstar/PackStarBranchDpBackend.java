package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.branchdp.BranchDpBackend;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.tools.MathTools;

/**
 * PACK* branch-DP backend.
 *
 * <p>The DP and sampling primitives live in the shared branch-DP engine, but
 * PACK* owns the runtime policy here: packstar.* config aliases,
 * no deterministic MARK* fallback, no branch lookahead/region-atom controls,
 * and a fixed-budget certified-or-abort estimator path.</p>
 */
final class PackStarBranchDpBackend extends BranchDpBackend implements AutoCloseable {

    private final String configuredSeedStateRole;
    private Integer calculationInstanceId = null;
    private PackStarEstimator activeEstimator = null;
    private int ccdSampleRecords = 0;
    private int smallStateNextRc = 0;

    PackStarBranchDpBackend(SimpleConfSpace confSpace,
                            EnergyMatrix rigidEmat,
                            EnergyMatrix minimizingEmat,
                            ConfEnergyCalculator minimizingConfEcalc,
                            RCs rcs,
                            Parallelism parallelism,
                            String stateNameOverride) {
        super(confSpace, rigidEmat, minimizingEmat, minimizingConfEcalc,
                rcs, parallelism, stateNameOverride);
        this.configuredSeedStateRole = stateNameOverride == null
                || stateNameOverride.trim().isEmpty()
                ? "standalone" : stateNameOverride.trim();
    }

    @Override
    public void init(double targetEpsilon) {
        super.init(targetEpsilon);
        smallStateNextRc = 0;
        ccdSampleRecords = 0;
        totalMinimizations = 0;
        if (!dpTablesReady) {
            flatSumZLower = java.math.BigDecimal.ZERO;
            flatSumZUpper = MathTools.BigPositiveInfinity;
            epsilonBound = 1.0;
        }
    }

    @Override
    protected void computeWithoutBranchDecomposition(int maxNumConfs) {
        if (rootedRoot == null
                && interactionGraph != null
                && interactionGraph.getNumPositions() <= 1) {

            if (!getStatus().canContinue() || maxNumConfs <= 0) return;
            int numPositions = searchRCs.getNumPos();
            // Zero positions has one empty conformation; one position can have many RCs.
            int numConformations = numPositions == 0 ? 1 : searchRCs.getNum(0);
            for (int computed = 0; computed < maxNumConfs
                    && smallStateNextRc < numConformations; computed++) {
                int[] conf = numPositions == 0 ? new int[0]
                        : new int[]{searchRCs.get(0, smallStateNextRc)};
                numConfsEnergied++;
                totalMinimizations++;
                ccdSampleRecords++;
                double energy = getMinimizingEcalc().calcEnergy(
                        new edu.duke.cs.osprey.confspace.RCTuple(conf)).energy;
                flatSumZLower = flatSumZLower.add(bc.calcPrecise(energy));
                smallStateNextRc++;
            }
            values.qstar = flatSumZLower;
            if (smallStateNextRc == numConformations) {
                flatSumZUpper = flatSumZLower;
                epsilonBound = 0.0;
                finishWithCurrentDPBounds("PACK*: exact small-state sum over "
                        + numConformations + " conformations");
            } else {
                // Unevaluated RCs still contribute: a partial sum is only a lower bound.
                values.pstar = MathTools.BigPositiveInfinity;
                values.qprime = MathTools.BigPositiveInfinity;
            }
            return;
        }
        abortPackStar("branch decomposition is unavailable; refusing to fall back to deterministic MARK*");
    }

    @Override
    protected boolean beforeBranchDecompositionSearch(int maxNumConfs) {
        if (!dpTablesReady) {
            abortPackStar("DP tables are unavailable; PACK* requires the DP/sampling backend");
            return true;
        }

        if (epsilonBound <= targetEpsilon) {

            finishWithCurrentDPBounds("PACK*: sampling skipped; first-round DP bounds already meet target, "
                    + "so CCD sampling and corrected DP are unnecessary. "
                    + "epsilon=" + String.format("%.6f", epsilonBound)
                    + " <= target=" + String.format("%.6f", targetEpsilon));
            return true;
        }

        computeWithPackStarEstimator(maxNumConfs);
        return true;
    }

    @Override
    public int getNumConfsEvaluated() {
        return activeEstimator == null ? super.getNumConfsEvaluated()
                : activeEstimator.getTotalCCDCalls();
    }

    @Override
    protected void attachExtraStats(PartitionFunction.Result result) {
        super.attachExtraStats(result);
        result.setStat(PackStarResult.FULL_CCD_STAT, getNumConfsEvaluated());
        result.setStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT, activeEstimator == null
                ? ccdSampleRecords : activeEstimator.getTotalCCDSampleRecords());
    }

    @Override
    public void setInstanceId(int val) {
        this.calculationInstanceId = val;
    }

    @Override
    public void close() {
        releaseLargeMemory();
    }

    private void abortPackStar(String reason) {

        System.out.println("PACK*: aborted: " + reason + ".");
        values = new edu.duke.cs.osprey.branchdp.BranchDpBackend.Values();
        values.qprime = MathTools.BigPositiveInfinity;
        setStatus(PartitionFunction.Status.Aborted);
    }

    private void computeWithPackStarEstimator(int sampleBudget) {
        System.out.println("PACK*: estimator activated for state=" + stateName
                + ", sampleBudget=" + (sampleBudget == Integer.MAX_VALUE ? "unbounded" : sampleBudget));

        PackStarEstimator estimator = new PackStarEstimator(
                rootedRoot, rootedRootEdge,
                branchMinimizingEmat, branchRigidEmat,
                interactionGraph, getMinimizingEcalc(),
                searchRCs,
                targetEpsilon,
                sampleBudget,
                randomStreamIdentity());

        double estimatorEpsilon;
        activeEstimator = estimator;
        try {
            estimatorEpsilon = estimator.compute();
        } finally {
            // Preserve work on successful, certificate-aborted, and exceptional exits.
            totalMinimizations = estimator.getTotalCCDCalls();
            numConfsEnergied = totalMinimizations;
            ccdSampleRecords = estimator.getTotalCCDSampleRecords();
            activeEstimator = null;
        }

        if (!estimator.hasValidCertificate()
                || !PackStarEstimator.isValidCertificate(
                estimator.getZLower(), estimator.getZUpper(), estimatorEpsilon)) {
            abortPackStar("estimator certificate is invalid: "
                    + estimator.getCertificateFailureReason());
            return;
        }

        PartitionFunction.Values vals = getValues();
        vals.qstar = estimator.getZLower();
        vals.pstar = estimator.getZUpper();
        vals.qprime = vals.pstar.subtract(vals.qstar);

        flatSumZLower = estimator.getZLower();
        flatSumZUpper = estimator.getZUpper();
        epsilonBound = estimatorEpsilon;

        System.out.println("PACK*: estimator finished. epsilon=" + String.format("%.6f", estimatorEpsilon)
                + ", CCD calls=" + estimator.getTotalCCDCalls()
                + ", cvPsi=" + String.format("%.4f", estimator.getCvPsi())
                + ", meanResidual=" + String.format("%.4f", estimator.getMeanResidual()) + " kcal/mol"
                + ", stdResidual=" + String.format("%.4f", estimator.getStdResidual()) + " kcal/mol");

        if (Double.isFinite(estimatorEpsilon)
                && estimatorEpsilon >= 0.0
                && estimatorEpsilon <= targetEpsilon) {
            setStatus(PartitionFunction.Status.Estimated);
        } else {
            System.out.println("PACK*: aborted: valid PAC interval missed target; epsilon="
                    + String.format("%.6f", estimatorEpsilon)
                    + " > target=" + String.format("%.6f", targetEpsilon)
                    + ", CCD calls=" + estimator.getTotalCCDCalls()
                    + ". No deterministic fallback or branch search was started.");
            setStatus(PartitionFunction.Status.Aborted);
        }
    }

    private String randomStreamIdentity() {
        String stateRole = calculationInstanceId == null
                ? configuredSeedStateRole
                : "kstar-state-" + calculationInstanceId;
        return PackStarConfig.stateKey(
                "packstar|" + stateRole + "|pac-v1", searchRCs);
    }
}
