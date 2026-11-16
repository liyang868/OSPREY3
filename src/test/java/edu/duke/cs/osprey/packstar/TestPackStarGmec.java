package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.confspace.RCTuple;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.confspace.Strand;
import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.ematrix.SimplerEnergyMatrixCalculator;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.energy.EnergyCalculator;
import edu.duke.cs.osprey.energy.forcefield.ForcefieldParams;
import edu.duke.cs.osprey.kstar.KStar;
import edu.duke.cs.osprey.kstar.KStarScore;
import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.structure.PDBIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

public class TestPackStarGmec {

    @TempDir Path output;

    @Test public void tracesTheActualFrozenLearnedProposal() throws Exception {
        Properties saved = (Properties) System.getProperties().clone();
        try {
            System.setProperty("packstar.dp.gpu", "false");
            System.setProperty("packstar.pac.sampling.gpu", "false");
            System.setProperty("packstar.pac.targetEpsilon", "0.99999");
            System.setProperty("packstar.pac.trainSamples", "128");
            System.setProperty("packstar.pac.maxEstSamples", "2048");
            Strand strand = new Strand.Builder(PDBIO.readFile("src/test/resources/1CC8.ss.pdb")).build();
            strand.flexibility.get("A16").setLibraryRotamers(Strand.WildType).setContinuous();
            strand.flexibility.get("A17").setLibraryRotamers(Strand.WildType).setContinuous();
            SimpleConfSpace space = new SimpleConfSpace.Builder().addStrand(strand).build();
            Parallelism parallelism = Parallelism.makeCpu(2);
            try (EnergyCalculator energy = new EnergyCalculator.Builder(space, new ForcefieldParams())
                    .setType(EnergyCalculator.Type.Cpu).setParallelism(parallelism).build()) {
                ConfEnergyCalculator calculator = new ConfEnergyCalculator.Builder(space, energy).build();
                EnergyMatrix matrix = new SimplerEnergyMatrixCalculator.Builder(calculator).build().calcEnergyMatrix();
                EnergyMatrix rigid = new EnergyMatrix(matrix);
                RCs allowed = new RCs(space);
                // A deliberately loose rigid upper-energy bound forces adaptation.
                for (int rc : allowed.get(0)) rigid.setOneBody(0, rc, rigid.getOneBody(0, rc) + 1000);
                try (var pf = new PackStarPartitionFunction(space, rigid, matrix, calculator, allowed, parallelism)) {
                    pf.init(0.99999);
                    pf.compute(Integer.MAX_VALUE);
                    PackStarResult ordinary = pf.makeResult();
                    assertNull(ordinary.gmec);
                    long calls = calculator.getNumCalculations();
                    pf.setGmecEnabled(true);
                    PackStarResult result = pf.makeResult();
                    assertNotNull(result.gmec, "The adaptive run must freeze a final proposal");
                    assertEquals(calls, calculator.getNumCalculations());
                    assertEquals(ordinary.numConfs, result.numConfs);
                    assertEquals(ordinary.values.calcLowerBound(), result.values.calcLowerBound());

                    var backendField = PackStarPartitionFunction.class.getDeclaredField("backend");
                    backendField.setAccessible(true);
                    Object backend = backendField.get(pf);
                    var proposalField = PackStarBranchDpBackend.class.getDeclaredField("finalProposalEmat");
                    proposalField.setAccessible(true);
                    EnergyMatrix proposal = (EnergyMatrix) proposalField.get(backend);
                    var baseField = edu.duke.cs.osprey.branchdp.BranchDpBackend.class
                            .getDeclaredField("branchMinimizingEmat");
                    baseField.setAccessible(true);
                    assertNotSame(baseField.get(backend), proposal);
                    double expected = Double.POSITIVE_INFINITY;
                    for (int a : allowed.get(0)) for (int b : allowed.get(1)) {
                        expected = Math.min(expected, proposal.getConstTerm() + proposal.getOneBody(0, a)
                                + proposal.getOneBody(1, b) + proposal.getPairwise(0, a, 1, b));
                    }
                    assertEquals(expected, result.gmec.getScore(), 1e-10);
                    int[] chosen = result.gmec.getAssignments();
                    assertEquals(expected, proposal.getConstTerm() + proposal.getOneBody(0, chosen[0])
                            + proposal.getOneBody(1, chosen[1])
                            + proposal.getPairwise(0, chosen[0], 1, chosen[1]), 1e-10);
                }
            }
        } finally {
            System.setProperties(saved);
        }
    }

    @Test public void optionalResultSurvivesCloseAndExportsEachSequencesStructure() throws Exception {
        Strand strand = new Strand.Builder(PDBIO.readFile("src/test/resources/1CC8.ss.pdb")).build();
        strand.flexibility.get("A16").setLibraryRotamers("ALA", "VAL").setContinuous();
        SimpleConfSpace space = new SimpleConfSpace.Builder().addStrand(strand).build();
        Parallelism parallelism = Parallelism.makeCpu(2);
        try (EnergyCalculator energy = new EnergyCalculator.Builder(space, new ForcefieldParams())
                .setType(EnergyCalculator.Type.Cpu).setParallelism(parallelism).build()) {
            ConfEnergyCalculator calculator = new ConfEnergyCalculator.Builder(space, energy).build();
            var matrix = new SimplerEnergyMatrixCalculator.Builder(calculator).build().calcEnergyMatrix();
            var sequences = new ArrayList<>(space.seqSpace.getMutants(1, true));
            if (space.seqSpace.containsWildTypeSequence()) sequences.add(space.seqSpace.makeWildTypeSequence());
            assertTrue(sequences.size() >= 2);
            List<KStar.ScoredSequence> scores = new ArrayList<>();
            for (var sequence : sequences) {
                var allowed = sequence.makeRCs(space);
                PackStarResult result;
                try (var pf = new PackStarPartitionFunction(space, matrix, matrix, calculator, allowed, parallelism)) {
                    pf.init(1e-6);
                    pf.compute(Integer.MAX_VALUE);
                    PackStarResult ordinary = pf.makeResult();
                    assertNull(ordinary.gmec);
                    long calls = calculator.getNumCalculations();
                    pf.setGmecEnabled(true);
                    result = pf.makeResult();
                    assertNotNull(result.gmec);
                    assertEquals(calls, calculator.getNumCalculations());
                    assertEquals(ordinary.numConfs, result.numConfs);
                    assertEquals(ordinary.values.calcLowerBound(), result.values.calcLowerBound());
                    assertSame(result.gmec, pf.makeResult().gmec);
                }
                double expected = Double.POSITIVE_INFINITY;
                for (int rc : allowed.get(0)) expected = Math.min(expected, matrix.getConstTerm() + matrix.getOneBody(0, rc));
                assertEquals(expected, result.gmec.getScore(), 1e-10);
                assertTrue(calculator.calcEnergy(new RCTuple(result.gmec.getAssignments())).energy < Double.POSITIVE_INFINITY);
                scores.add(new KStar.ScoredSequence(sequence, new KStarScore(result, result, result)));
            }
            assertFalse(Files.exists(output.resolve("test_gmec.csv")));
            PackStarDesignRunner.writeGmecResults(scores, calculator, "test", output.toString());
            List<String> rows = Files.readAllLines(output.resolve("test_gmec.csv"));
            assertEquals(scores.size() + 1, rows.size());
            assertTrue(rows.get(0).contains("proposal_gmec_energy_kcal_mol"));
            for (int i = 0; i < scores.size(); i++) {
                assertTrue(rows.get(i + 1).contains(",EXPORTED,"));
                Path pdb = output.resolve(String.format("test_gmec/rank_%04d.pdb", i + 1));
                assertTrue(Files.isRegularFile(pdb));
                var molecule = PDBIO.readFile(pdb.toString());
                var gmec = ((PackStarResult) scores.get(i).score.complex).gmec;
                String expectedType = space.positions.get(0).resConfs.get(gmec.getAssignments()[0]).template.name;
                assertTrue(molecule.getResByPDBResNumber("A16").fullName.startsWith(expectedType));
                assertTrue(Files.readString(pdb).contains("final proposal GMEC"));
            }
        }
    }
}
