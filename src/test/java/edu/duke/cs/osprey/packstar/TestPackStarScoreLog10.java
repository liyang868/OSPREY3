package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.kstar.KStarScore;
import edu.duke.cs.osprey.kstar.KStar;

import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.tools.MathTools;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import edu.duke.cs.osprey.confspace.SeqSpace;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.Duration;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class TestPackStarScoreLog10 {

    @Test
    public void extremeOutputFinishesQuickly() {
        assertTimeoutPreemptively(Duration.ofSeconds(2), () -> {
            String[] values = {"4.434671e-11034", "8.052576e-48864", "3.721791e-86822",
                    "9.399761e-86822", "1e-289233", "7.25e289233"};
            double[] expected = {-11033.3531385958, -48863.0940651656, -86821.4292480177,
                    -86821.0268831900, -289233, 289233.8603380066};
            for (int i = 0; i < values.length; i++) {
                assertEquals(expected[i], KStarScore.scoreToLog10(new BigDecimal(values[i])), 1e-6);
            }
        });
    }

    @Test
    public void scaleLimitsDoNotOverflow() {
        assertEquals(-2147483647.0, KStarScore.scoreToLog10(new BigDecimal(BigInteger.ONE, Integer.MAX_VALUE)));
        assertEquals(2147483648.0, KStarScore.scoreToLog10(new BigDecimal(BigInteger.ONE, Integer.MIN_VALUE)));
        assertEquals(2147483650.0, KStarScore.scoreToLog10(new BigDecimal(BigInteger.valueOf(100), Integer.MIN_VALUE)));
    }

    @Test
    public void ordinaryValuesKeepExistingPrecision() {
        for (String value : new String[]{"1", "10", "0.001", "3.699649e103", "2.125e-308",
                "1e308", "1.00000000000000000001", "0.99999999999999999999"}) {
            BigDecimal x = new BigDecimal(value);
            assertEquals(MathTools.log10(x), KStarScore.scoreToLog10(x));
        }
        for (String value : new String[]{"1.75e-309", "8.625e309"}) {
            BigDecimal x = new BigDecimal(value);
            assertEquals(MathTools.log10(x), KStarScore.scoreToLog10(x), 1e-12);
        }
    }

    @Test
    public void specialValuesKeepExistingMeaning() {
        assertNull(KStarScore.scoreToLog10(null));
        assertEquals("none", KStarScore.scoreToLog10String(null));
        assertEquals(Double.NEGATIVE_INFINITY, KStarScore.scoreToLog10(new BigDecimal("0e-86822")));
        assertEquals(Double.POSITIVE_INFINITY, KStarScore.scoreToLog10(MathTools.BigPositiveInfinity));
        assertTrue(KStarScore.scoreToLog10(MathTools.BigNaN).isNaN());
        assertTrue(KStarScore.scoreToLog10(MathTools.BigNegativeInfinity).isNaN());
        assertTrue(KStarScore.scoreToLog10(new BigDecimal("-1e-86822")).isNaN());
    }

    @Test
    public void tinyPartitionFunctionsRemainEstimated() {
        PartitionFunction.Values values = new PartitionFunction.Values();
        BigDecimal z = new BigDecimal("3.721791e-86822");
        values.qstar = z;
        values.qprime = BigDecimal.ZERO;
        values.pstar = BigDecimal.ZERO;
        PartitionFunction.Result result = new PartitionFunction.Result(PartitionFunction.Status.Estimated, values, 1);
        KStarScore score = new KStarScore(result, result, result);
        assertNotNull(score.score);
        assertTrue(Double.isFinite(score.scoreLog10()));
        assertTrue(Double.isFinite(score.lowerBoundLog10()));
        assertTrue(Double.isFinite(score.upperBoundLog10()));
        assertSame(z, values.qstar);
        assertEquals(PartitionFunction.Status.Estimated, result.status);
    }

    @Test
    public void productionCsvSortAndFormattingFinishQuickly(@TempDir Path output) {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            var constructor = SeqSpace.class.getDeclaredConstructor();
            constructor.setAccessible(true);
            var sequence = constructor.newInstance().makeWildTypeSequence();
            var scores = new ArrayList<KStar.ScoredSequence>();
            for (int exponent : new int[]{-86822, 86822, -11034, -48864, -289233, 289233}) {
                var protein = result(BigDecimal.ONE);
                var ligand = result(BigDecimal.ONE);
                var complex = result(new BigDecimal("3.721791e" + exponent));
                scores.add(new KStar.ScoredSequence(sequence, new KStarScore(protein, ligand, complex)));
            }
            var writer = PackStarDesignRunner.class.getDeclaredMethod("writeKStarResults", List.class,
                    String.class, String.class, String.class, double.class, double.class);
            writer.setAccessible(true);
            writer.invoke(null, scores, "extreme", "packstar", output.toString(), 0.683, 12.3);
            List<String> lines = Files.readAllLines(output.resolve("extreme_packstar.csv"));
            assertEquals(scores.size() + 1, lines.size());
            List<String> header = Arrays.asList(lines.get(0).split(",", -1));
            assertEquals(29, header.size());
            for (int i = 0; i < scores.size(); i++) {
                if (i > 0) assertTrue(scores.get(i - 1).score.lowerBound.compareTo(scores.get(i).score.lowerBound) > 0);
                String[] columns = lines.get(i + 1).split(",", -1);
                assertEquals(header.size(), columns.length);
                assertEquals(Integer.toString(i + 1), columns[0]);
                assertTrue(Double.isFinite(Double.parseDouble(columns[4])));
                for (String state : new String[]{"prot", "lig", "comp"}) {
                    assertEquals("Estimated", columns[header.indexOf(state + "_status")]);
                }
                assertEquals(scores.get(i).score.complex.status, PartitionFunction.Status.Estimated);
            }
        });
    }

    private static PartitionFunction.Result result(BigDecimal z) {
        var values = new PartitionFunction.Values();
        values.qstar = z;
        values.qprime = BigDecimal.ZERO;
        values.pstar = BigDecimal.ZERO;
        return new PartitionFunction.Result(PartitionFunction.Status.Estimated, values, 1);
    }
}
