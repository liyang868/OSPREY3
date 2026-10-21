package edu.duke.cs.osprey.packstar;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class TestPackStarProposalLearning {
    @Test void jointFitCorrectsTheSumRatherThanEachTermIndependently() {
        int[][] features = {{0, 2}, {0, 3}, {1, 2}, {1, 3}};
        double[] residual = {1, 3, 4, 6};
        var fit = PackStarProposalLearning.jointFit(features, residual,
                new double[4], new double[4], 10, 100);
        for (int i = 0; i < 4; i++) {
            double predicted = fit.offset;
            for (int j : features[i]) predicted += fit.delta[j];
            assertEquals(residual[i], predicted, 1e-7);
        }
    }

    @Test void jointFitPreservesMultiplicityAndSourceWeights() {
        var weighted = PackStarProposalLearning.jointFit(
                new int[][]{{}, {}}, new double[]{0, 4}, new double[]{Math.log(3), 0},
                new double[0], 10, 10);
        var repeated = PackStarProposalLearning.jointFit(
                new int[][]{{}, {}, {}, {}}, new double[]{0, 0, 0, 4}, new double[4],
                new double[0], 10, 10);
        assertEquals(1, weighted.offset, 1e-12);
        assertEquals(weighted.offset, repeated.offset, 1e-12);
    }

    @Test void modelSelectionUsesTheGlobalSampleMinimum() {
        record Model(long samples, int complexity) {}
        var simpler = Comparator.comparingInt(Model::complexity);
        Model best = new Model(100, 3), middle = new Model(105, 2), outside = new Model(110, 0);
        List<Model> order = new ArrayList<>(List.of(best, middle, outside));
        for (int seed = 0; seed < 6; seed++) {
            Collections.shuffle(order, new Random(seed));
            assertSame(middle, PackStarProposalLearning.chooseWithinFivePercent(
                    order, Model::samples, simpler));
        }
        assertNull(PackStarProposalLearning.chooseWithinFivePercent(
                List.<Model>of(), Model::samples, simpler));
    }
}
