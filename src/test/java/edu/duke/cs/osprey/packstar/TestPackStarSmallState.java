package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.confspace.RCTuple;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.confspace.Strand;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.ematrix.SimplerEnergyMatrixCalculator;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.energy.EnergyCalculator;
import edu.duke.cs.osprey.energy.forcefield.ForcefieldParams;
import edu.duke.cs.osprey.kstar.pfunc.BoltzmannCalculator;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.structure.PDBIO;
import edu.duke.cs.osprey.tools.MathTools;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

/** End-to-end checks of the no-decomposition path against full CCD enumeration. */
public class TestPackStarSmallState {

    private record Fixture(SimpleConfSpace space, ConfEnergyCalculator calculator,
                           EnergyMatrix matrix, Parallelism parallelism, RCs rcs) {
        PackStarPartitionFunction pfunc(RCs allowed) {
            return new PackStarPartitionFunction(space, matrix, matrix, calculator, allowed, parallelism);
        }

        BigDecimal exhaustiveZ(RCs allowed) {
            var bc = new BoltzmannCalculator(PartitionFunction.decimalPrecision);
            if (allowed.getNumPos() == 0)
                return bc.calcPrecise(calculator.calcEnergy(new RCTuple(new int[0])).energy);
            BigDecimal z = BigDecimal.ZERO;
            for (int rc : allowed.get(0))
                z = z.add(bc.calcPrecise(calculator.calcEnergy(new RCTuple(new int[]{rc})).energy));
            return z;
        }
    }

    @FunctionalInterface
    private interface Check { void run(Fixture fixture) throws Exception; }

    private void withSpace(int positions, Check check) throws Exception {
        Properties saved = (Properties) System.getProperties().clone();
        try {
            System.setProperty("packstar.dp.gpu", "false");
            Strand strand = new Strand.Builder(PDBIO.readFile("src/test/resources/1CC8.ss.pdb")).build();
            if (positions == 1)
                strand.flexibility.get("A16").setLibraryRotamers(Strand.WildType)
                        .addWildTypeRotamers().setContinuous();
            var space = new SimpleConfSpace.Builder().addStrand(strand).build();
            var parallelism = Parallelism.makeCpu(2);
            try (var energy = new EnergyCalculator.Builder(space, new ForcefieldParams())
                    .setType(EnergyCalculator.Type.Cpu).setIsMinimizing(true)
                    .setParallelism(parallelism).build()) {
                var calculator = new ConfEnergyCalculator.Builder(space, energy).build();
                var matrix = new SimplerEnergyMatrixCalculator.Builder(calculator).build().calcEnergyMatrix();
                check.run(new Fixture(space, calculator, matrix, parallelism, new RCs(space)));
            }
        } finally {
            System.setProperties(saved);
        }
    }

    private static void assertExact(PackStarPartitionFunction pf, BigDecimal expected, int calls) {
        var result = pf.makeResult();
        assertEquals(PartitionFunction.Status.Estimated, result.status);
        BigDecimal lower = result.values.calcLowerBound();
        assertEquals(lower, result.values.calcUpperBound());
        if (expected.signum() == 0) assertEquals(0, lower.signum());
        else assertEquals(1.0, lower.divide(expected, PartitionFunction.decimalPrecision).doubleValue(), 1e-12);
        assertEquals(0.0, result.values.getEffectiveEpsilon());
        assertEquals(calls, result.numConfs);
        assertEquals(calls, result.getStat(PackStarResult.FULL_CCD_STAT));
        assertEquals(calls, result.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT));
    }

    @Test public void sumsEveryAllowedRotamer() throws Exception {
        withSpace(1, f -> {
            assertTrue(f.rcs.getNum(0) > 1);
            BigDecimal expected = f.exhaustiveZ(f.rcs);
            long before = f.calculator.getNumCalculations();
            try (var pf = f.pfunc(f.rcs)) {
                pf.init(1e-12);
                pf.compute(Integer.MAX_VALUE);
                assertExact(pf, expected, f.rcs.getNum(0));
                assertEquals(f.rcs.getNum(0), f.calculator.getNumCalculations() - before);
            }
        });
    }

    @Test public void respectsNoncontiguousAllowedRcIndices() throws Exception {
        withSpace(1, f -> {
            var allowed = new RCs(new int[][]{{2, 4, f.rcs.get(0, f.rcs.getNum(0) - 1)}});
            BigDecimal expected = f.exhaustiveZ(allowed);
            try (var pf = f.pfunc(allowed)) {
                pf.init(1e-12);
                pf.compute(Integer.MAX_VALUE);
                assertExact(pf, expected, 3);
            }
        });
    }

    @Test public void boundedCallsResumeWithoutIssuingAnIncompleteCertificate() throws Exception {
        withSpace(1, f -> {
            BigDecimal expected = f.exhaustiveZ(f.rcs);
            try (var pf = f.pfunc(f.rcs)) {
                pf.init(1e-12);
                pf.compute(0);
                assertEquals(0, pf.getNumConfsEvaluated());
                pf.compute(1);
                assertEquals(1, pf.getNumConfsEvaluated());
                assertEquals(PartitionFunction.Status.Estimating, pf.getStatus());
                assertSame(MathTools.BigPositiveInfinity, pf.getValues().calcUpperBound());
                assertEquals(1.0, pf.getValues().getEffectiveEpsilon());
                pf.compute(2);
                assertEquals(3, pf.getNumConfsEvaluated());
                assertEquals(PartitionFunction.Status.Estimating, pf.getStatus());
                pf.compute(Integer.MAX_VALUE);
                assertExact(pf, expected, f.rcs.getNum(0));
                pf.compute(Integer.MAX_VALUE);
                assertExact(pf, expected, f.rcs.getNum(0));
                pf.init(1e-12);
                assertEquals(0, pf.getNumConfsEvaluated());
                pf.compute(Integer.MAX_VALUE);
                assertExact(pf, expected, f.rcs.getNum(0));
            }
        });
    }

    @Test public void oneAllowedRcUsesItsGlobalIndex() throws Exception {
        withSpace(1, f -> {
            var allowed = new RCs(new int[][]{{f.rcs.get(0, f.rcs.getNum(0) - 1)}});
            BigDecimal expected = f.exhaustiveZ(allowed);
            try (var pf = f.pfunc(allowed)) {
                pf.init(1e-12);
                pf.compute(1);
                assertExact(pf, expected, 1);
            }
        });
    }

    @Test public void noPositionsStillHasTheEmptyConformation() throws Exception {
        withSpace(0, f -> {
            BigDecimal expected = f.exhaustiveZ(f.rcs);
            try (var pf = f.pfunc(f.rcs)) {
                pf.init(1e-12);
                pf.compute(1);
                assertExact(pf, expected, 1);
            }
        });
    }

    @Test public void noAllowedRcsHasZeroPartitionFunction() throws Exception {
        withSpace(1, f -> {
            try (var pf = f.pfunc(new RCs(new int[][]{new int[0]}))) {
                pf.init(1e-12);
                pf.compute(Integer.MAX_VALUE);
                assertExact(pf, BigDecimal.ZERO, 0);
            }
        });
    }
}
