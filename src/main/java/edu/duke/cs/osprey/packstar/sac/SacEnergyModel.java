/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.sac;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.branchdp.BranchDecomposition;
import edu.duke.cs.osprey.branchdp.BranchDpBackend;
import edu.duke.cs.osprey.branchdp.InteractionGraph;
import edu.duke.cs.osprey.branchdp.RootedTreeEdge;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.kstar.pfunc.BoltzmannCalculator;

import java.util.*;

/** The three uncut, pairwise Em partition functions used by SAC. All scores are natural logarithms. */
final class SacEnergyModel {
    static final double RT = BoltzmannCalculator.RClassic * BoltzmannCalculator.TClassic;
    final String[] names;
    final String[][] alphabet;
    final int[] reference;
    final long sequenceCount;
    final State[] states;

    SacEnergyModel(SimpleConfSpace[] spaces, EnergyMatrix[] matrices) {
        SimpleConfSpace complex = spaces[0];
        names = complex.positions.stream().map(p -> p.resNum).toArray(String[]::new);
        alphabet = new String[names.length][];
        reference = new int[names.length];
        long count = 1;
        for (var p : complex.positions) {
            alphabet[p.index] = p.resConfs.stream().map(rc -> rc.template.name.toUpperCase(Locale.ROOT))
                .distinct().sorted().toArray(String[]::new);
            reference[p.index] = Arrays.asList(alphabet[p.index]).indexOf(p.resFlex.wildType.toUpperCase(Locale.ROOT));
            if (reference[p.index] < 0) throw new IllegalArgumentException("Wild type missing at " + p.resNum);
            count = Math.min(Integer.MAX_VALUE + 1L, count * alphabet[p.index].length);
        }
        sequenceCount = count;
        states = new State[3];
        BitSet unbound = new BitSet();
        for (int s = 0; s < states.length; s++) {
            int[] global = new int[spaces[s].positions.size()];
            int[][][] domains = new int[global.length][][];
            for (var p : spaces[s].positions) {
                int i = complex.getPositionOrThrow(p.resNum).index;
                if (s > 0 && unbound.get(i)) throw new IllegalArgumentException("Overlapping unbound position " + p.resNum);
                if (s > 0) unbound.set(i);
                global[p.index] = i;
                domains[p.index] = new int[alphabet[i].length][];
                for (int a = 0; a < alphabet[i].length; a++) {
                    String type = alphabet[i][a];
                    int[] rows = p.resConfs.stream().filter(rc -> rc.template.name.equalsIgnoreCase(type))
                        .mapToInt(rc -> rc.index).toArray();
                    if (rows.length == 0) throw new IllegalArgumentException("Missing " + type + " RCs at " + p.resNum);
                    domains[p.index][a] = rows;
                }
                if (Arrays.stream(domains[p.index]).mapToInt(rows -> rows.length).sum() != p.resConfs.size()) {
                    throw new IllegalArgumentException("State alphabets differ at " + p.resNum);
                }
            }
            states[s] = new State(matrices[s], global, domains);
        }
        if (unbound.cardinality() != names.length) throw new IllegalArgumentException("Unbound states must cover the complex");
    }

    static final class Key {
        final int[] values;
        private final int hash;
        Key(int[] values) { this.values = values.clone(); hash = Arrays.hashCode(values); }
        @Override public int hashCode() { return hash; }
        @Override public boolean equals(Object other) { return other instanceof Key k && Arrays.equals(values, k.values); }
    }

    record Region(BitSet inside, BitSet boundary) {
        BitSet variables() {
            BitSet all = (BitSet)inside.clone();
            all.or(boundary);
            return all;
        }
    }
    private record MessageKey(Region region, Key sequence) {}
    private record Plan(BranchDecomposition tree, int split, double work, List<Region> regions) {}

    static final class State {
        final EnergyMatrix matrix;
        final int[] global;
        final int[][][] domains; // state position, sequence allele, original RC indices
        final BitSet all;
        final BitSet[] neighbors;
        final Region whole;
        private final InteractionGraph graph;
        private final Map<Key, Plan> plans = new LinkedHashMap<>(16, .75f, true);

        State(EnergyMatrix matrix, int[] global, int[][][] domains) {
            this.matrix = matrix;
            this.global = global;
            this.domains = domains;
            if (matrix.hasHigherOrderTerms() || matrix.hasHigherOrderTuples()) throw new IllegalArgumentException("SAC requires pairwise Em");
            if (!Double.isFinite(matrix.getConstTerm())) throw new IllegalArgumentException("Nonfinite Em constant");
            graph = InteractionGraph.buildComplete(global.length);
            all = new BitSet();
            neighbors = new BitSet[global.length];
            for (int p = 0; p < global.length; p++) {
                all.set(p);
                neighbors[p] = new BitSet();
                for (int a = 0; a < matrix.getNumConfAtPos(p); a++) checkEnergy(matrix.getOneBody(p, a));
                for (int q = 0; q < p; q++) {
                    boolean coupled = false;
                    for (int a = 0; a < matrix.getNumConfAtPos(p); a++) {
                        for (int b = 0; b < matrix.getNumConfAtPos(q); b++) {
                            double e = matrix.getPairwise(p, a, q, b);
                            checkEnergy(e);
                            coupled |= e != 0;
                        }
                    }
                    if (coupled) { neighbors[p].set(q); neighbors[q].set(p); }
                }
            }
            whole = new Region(all, new BitSet());
        }

        private static void checkEnergy(double energy) {
            if (Double.isNaN(energy) || energy == Double.NEGATIVE_INFINITY) throw new IllegalArgumentException("Invalid Em energy");
        }

        int[][] rows(int[] seq) {
            int[][] rows = new int[global.length][];
            for (int p = 0; p < rows.length; p++) rows[p] = domains[p][seq[global[p]]];
            return rows;
        }

        Key projection(BitSet positions, int[] seq) {
            return new Key(positions.stream().map(p -> seq[global[p]]).toArray());
        }

        private Plan plan(int[][] rows) {
            int[] sizes = Arrays.stream(rows).mapToInt(r -> r.length).toArray();
            Key shape = new Key(sizes);
            Plan plan = plans.get(shape);
            if (plan != null) return plan;
            var tree = new BranchDecomposition(graph, BranchDecomposition.Strategy.WEIGHTED_HICKS, sizes, false);
            tree.compute();
            var selected = BranchDpBackend.selectConfiguredRoot(tree, new RCs(rows), false);
            if (selected == null) throw new IllegalStateException("Missing SAC Em decomposition root");
            try {
                List<RootedTreeEdge> edges = new ArrayList<>();
                RootedTreeEdge.collectLambdaEdges(selected.root, edges);
                List<Region> regions = new ArrayList<>();
                double work = 0;
                for (var edge : edges) {
                    BitSet inside = new BitSet(), boundary = new BitSet();
                    edge.getL().forEach(inside::set);
                    edge.getM().forEach(boundary::set);
                    regions.add(new Region(inside, boundary));
                    work += (double)edge.getMStateCount() * edge.getTotalLambdaStates();
                }
                plan = new Plan(tree, selected.splitEdgeIndex, work, regions);
                if (plans.size() == 128) plans.remove(plans.keySet().iterator().next());
                plans.put(shape, plan);
                return plan;
            } finally {
                RootedTreeEdge.postOrderReleaseLargeMemory(selected.root);
            }
        }
    }

    /** Numeric caches belong to a rollout. Copies of a warm cache share immutable tables. */
    static final class Cache {
        private static final int MAX_ENTRIES = 2_000_000;
        final State state;
        private final Map<MessageKey, double[]> messages = new LinkedHashMap<>(32, .75f, true);
        private final Map<Key, Double> scores = new LinkedHashMap<>(32, .75f, true);
        private int entries;
        double work;

        Cache(State state) { this.state = state; }
        Cache(Cache warm) {
            this(warm.state);
            messages.putAll(warm.messages);
            scores.putAll(warm.scores);
            entries = warm.entries;
        }
        private MessageKey key(Region region, int[] seq) { return new MessageKey(region, state.projection(region.variables(), seq)); }
        double[] message(Region region, int[] seq) { return messages.get(key(region, seq)); }
        private void save(MessageKey key, double[] values) {
            if (values.length > MAX_ENTRIES) return;
            double[] old = messages.remove(key);
            if (old != null) entries -= old.length;
            while (entries + values.length > MAX_ENTRIES) {
                var it = messages.values().iterator();
                entries -= it.next().length;
                it.remove();
            }
            messages.put(key, values);
            entries += values.length;
        }

        double estimate(int[] seq) {
            if (scores.containsKey(state.projection(state.all, seq))) return 0;
            int[][] rows = state.rows(seq);
            if (rows.length > 2) return state.plan(rows).work;
            return Arrays.stream(rows).mapToDouble(r -> r.length).reduce(1, (a, b) -> a * b);
        }

        double score(int[] seq, Runnable check) {
            Key key = state.projection(state.all, seq);
            Double old = scores.get(key);
            if (old != null) return old;
            int[][] rows = state.rows(seq);
            double logZ = rows.length > 2 ? fullDp(seq, rows, check) : smallState(rows, check);
            logZ -= state.matrix.getConstTerm() / RT;
            if (Double.isNaN(logZ) || logZ == Double.POSITIVE_INFINITY) throw new ArithmeticException("Invalid SAC log Z");
            if (scores.size() == 100_000) scores.remove(scores.keySet().iterator().next());
            scores.put(key, logZ);
            return logZ;
        }

        private double smallState(int[][] rows, Runnable check) {
            if (rows.length == 0) return 0;
            double sum = Double.NEGATIVE_INFINITY;
            for (int a : rows[0]) {
                check.run();
                double u = state.matrix.getOneBody(0, a);
                if (rows.length == 1) {
                    sum = RootedTreeEdge.logSumExp(sum, -u / RT);
                    work++;
                } else for (int b : rows[1]) {
                    double energy = u + state.matrix.getOneBody(1, b) + state.matrix.getPairwise(0, a, 1, b);
                    sum = RootedTreeEdge.logSumExp(sum, -energy / RT);
                    work++;
                }
            }
            return sum;
        }

        private double fullDp(int[] seq, int[][] rows, Runnable check) {
            Plan plan = state.plan(rows);
            var root = plan.tree.rootBranchTree(new RCs(rows), plan.split);
            try {
                RootedTreeEdge.postOrderCompLlambda(root, true);
                var rootEdge = root.getLeftChild().getChildOfEdge();
                rootEdge.compactTree();
                List<RootedTreeEdge> edges = new ArrayList<>();
                RootedTreeEdge.collectLambdaEdges(root, edges);
                for (int i = 0; i < edges.size(); i++) {
                    check.run();
                    var edge = edges.get(i);
                    MessageKey key = key(plan.regions.get(i), seq);
                    double[] table = messages.get(key);
                    if (table != null) {
                        for (int j = 0; j < table.length; j++) edge.setLogZ(j, table[j], table[j]);
                        continue;
                    }
                    edge.initIncrementalEnumeration(state.matrix, state.matrix, state.graph, RT, false);
                    edge.computeFullDP();
                    long count = edge.getMStateCount();
                    work += (double)count * edge.getTotalLambdaStates();
                    if (count <= MAX_ENTRIES) {
                        table = new double[(int)count];
                        for (int j = 0; j < count; j++) table[j] = edge.getLogZUpper(j);
                        save(key, table);
                    }
                }
                return rootEdge.getLogZUpper(0);
            } finally {
                RootedTreeEdge.postOrderReleaseLargeMemory(root);
            }
        }

        // Prefer the factors incident to the mutation. For a large boundary use a common
        // native subtree, whose message is conditioned on the same ordered boundary RCs.
        Region response(int[] source, int[] target, Runnable check) {
            BitSet changed = new BitSet(), boundary = new BitSet();
            for (int p = 0; p < state.global.length; p++) {
                int i = state.global[p];
                if (source[i] != target[i]) { changed.set(p); boundary.or(state.neighbors[p]); }
            }
            if (changed.isEmpty()) return null;
            boundary.andNot(changed);
            Region local = new Region(changed, boundary);
            int[][] before = state.rows(source), after = state.rows(target);
            if (count(boundary, before) <= 100_000 && count(local.variables(), before) <= 2_000_000
                && count(local.variables(), after) <= 2_000_000) {
                localTable(local, source, before, check);
                localTable(local, target, after, check);
                return local;
            }
            Region best = state.whole;
            if (state.global.length > 2) {
                for (Region r : state.plan(before).regions) {
                    BitSet missing = (BitSet)changed.clone();
                    missing.andNot(r.inside);
                    if (missing.isEmpty() && r.variables().cardinality() < best.variables().cardinality()
                        && message(r, source) != null && message(r, target) != null) best = r;
                }
            }
            return best;
        }

        private static double count(BitSet positions, int[][] rows) {
            return positions.stream().mapToDouble(p -> rows[p].length).reduce(1, (a, b) -> a * b);
        }

        private void localTable(Region region, int[] seq, int[][] rows, Runnable check) {
            MessageKey key = key(region, seq);
            if (messages.containsKey(key)) return;
            int[] boundary = region.boundary.stream().toArray(), inside = region.inside.stream().toArray();
            int[] rc = new int[rows.length];
            double[] table = new double[(int)count(region.boundary, rows)];
            int terms = (int)count(region.inside, rows);
            for (int j = 0; j < table.length; j++) {
                decode(j, boundary, rows, rc);
                double sum = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < terms; k++) {
                    if ((k & 1023) == 0) check.run();
                    decode(k, inside, rows, rc);
                    double e = 0;
                    for (int u = 0; u < inside.length; u++) {
                        int p = inside[u];
                        e += state.matrix.getOneBody(p, rc[p]);
                        for (int v = 0; v < u; v++) e += state.matrix.getPairwise(p, rc[p], inside[v], rc[inside[v]]);
                        for (int q : boundary) e += state.matrix.getPairwise(p, rc[p], q, rc[q]);
                    }
                    sum = RootedTreeEdge.logSumExp(sum, -e / RT);
                }
                table[j] = sum;
                work += terms;
            }
            save(key, table);
        }

        private static void decode(int index, int[] axes, int[][] rows, int[] rc) {
            for (int j = axes.length - 1; j >= 0; j--) {
                int p = axes[j];
                rc[p] = rows[p][index % rows[p].length];
                index /= rows[p].length;
            }
        }
    }
}
