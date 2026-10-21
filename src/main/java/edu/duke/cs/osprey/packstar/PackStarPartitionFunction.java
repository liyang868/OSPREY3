package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.branchdp.BranchDpConfig;
import edu.duke.cs.osprey.confspace.ConfDB;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.parallelism.Parallelism;

import java.math.BigDecimal;
import java.util.function.Supplier;

/**
 * PACK* partition-function entry point.
 *
 * <p>Calls into the shared branch-DP engine execute in the PACK* configuration
 * scope, including construction and cleanup.</p>
 */
public class PackStarPartitionFunction implements PartitionFunction.WithConfDB, AutoCloseable {

    private final PackStarBranchDpBackend backend;

    public PackStarPartitionFunction(SimpleConfSpace confSpace,
                                     EnergyMatrix rigidEmat,
                                     EnergyMatrix minimizingEmat,
                                     ConfEnergyCalculator minimizingConfEcalc,
                                     RCs rcs,
                                     Parallelism parallelism) {
        this(confSpace, rigidEmat, minimizingEmat, minimizingConfEcalc, rcs, parallelism, null);
    }

    public PackStarPartitionFunction(SimpleConfSpace confSpace,
                                     EnergyMatrix rigidEmat,
                                     EnergyMatrix minimizingEmat,
                                     ConfEnergyCalculator minimizingConfEcalc,
                                     RCs rcs,
                                     Parallelism parallelism,
                                     String stateNameOverride) {
        try (BranchDpConfig.Scope scope = BranchDpConfig.enterPackStarAliasScope()) {
            backend = new PackStarBranchDpBackend(confSpace, rigidEmat, minimizingEmat,
                    minimizingConfEcalc, rcs, parallelism, stateNameOverride);
        }
    }

    @Override
    public void setInstanceId(int val) {
        runInPackStarBackend(() -> backend.setInstanceId(val));
    }

    @Override
    public void setReportProgress(boolean val) {
        runInPackStarBackend(() -> backend.setReportProgress(val));
    }

    @Override
    public void setConfListener(ConfListener val) {
        runInPackStarBackend(() -> backend.setConfListener(val));
    }

    @Override
    public void init(double targetEpsilon) {
        runInPackStarBackend(() -> backend.init(targetEpsilon));
    }

    @Override
    public void setStabilityThreshold(BigDecimal stabilityThreshold) {
        runInPackStarBackend(() -> backend.setStabilityThreshold(stabilityThreshold));
    }

    @Override
    public Status getStatus() {
        return callInPackStarBackend(backend::getStatus);
    }

    @Override
    public Values getValues() {
        return callInPackStarBackend(backend::getValues);
    }

    @Override
    public int getParallelism() {
        return callInPackStarBackend(backend::getParallelism);
    }

    @Override
    public int getNumConfsEvaluated() {
        return callInPackStarBackend(backend::getNumConfsEvaluated);
    }

    @Override
    public void compute(int maxNumConfs) {
        runInPackStarBackend(() -> backend.compute(maxNumConfs));
    }

    @Override
    public PackStarResult makeResult() {
        return new PackStarResult(callInPackStarBackend(backend::makeResult));
    }

    @Override
    public void setConfDB(ConfDB confDB, ConfDB.Key key) {
        runInPackStarBackend(() -> backend.setConfDB(confDB, key));
    }

    @Override
    public void close() {
        runInPackStarBackend(backend::close);
    }

    private void runInPackStarBackend(Runnable action) {
        try (BranchDpConfig.Scope scope = BranchDpConfig.enterPackStarAliasScope()) {
            action.run();
        }
    }

    private <T> T callInPackStarBackend(Supplier<T> supplier) {
        try (BranchDpConfig.Scope scope = BranchDpConfig.enterPackStarAliasScope()) {
            return supplier.get();
        }
    }
}
