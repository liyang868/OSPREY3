/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.search;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.duke.cs.osprey.branchdp.BranchDpBackend.RootSelectionException;
import edu.duke.cs.osprey.branchdp.DPTableTooLargeException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static edu.duke.cs.osprey.packstar.search.EmEnergyModel.*;

/** Shared candidates, Em scores and query accounting for the sequence search routes. */
final class EmSequenceSearch {
    static final ObjectMapper JSON = new ObjectMapper();

    record Row(int[] seq, double[] logZ, double score) {}
    private record Motif(int[] positions, Row target, double gain, double synergy) {}
    enum Kind { LOCAL, EXPAND, REMOTE, BATCH }

    record Mutation(Row source, int[] positions, Row target, BitSet pinned, Region[] regions, double lower, double upper) {
        boolean compatible(int[] background) {
            for (int i = pinned.nextSetBit(0); i >= 0; i = pinned.nextSetBit(i + 1)) {
                if (source.seq[i] != background[i]) return false;
            }
            return true;
        }
        boolean bounded() { return Double.isFinite(lower) && Double.isFinite(upper); }
        double width() { return upper - lower; }
    }

    static final class Action {
        final Kind kind;
        final Row background;
        final int[] positions;
        final List<int[]> candidates;
        final double hint, synergy;
        Mutation match, evidence;
        boolean batchOnly;

        Action(Kind kind, Row background, int[] positions, List<int[]> candidates, double hint, double synergy) {
            this.kind = kind;
            this.background = background;
            this.positions = positions;
            this.candidates = candidates;
            this.hint = hint;
            this.synergy = synergy;
        }
    }

    private static int[] changed(int[] a, int[] b) {
        return IntStream.range(0, a.length).filter(i -> a[i] != b[i]).toArray();
    }

    static int distance(int[] a, int[] b) {
        int d = 0;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) d++;
        return d;
    }

    static final class Horizon extends RuntimeException {}

    static final int POOL = 64;
    final EmEnergyModel model;
    final Cache[] caches;
    final Map<Key, Row> rows = new LinkedHashMap<>();
    final Set<Key> failed = new HashSet<>();
    final List<Mutation> mutations = new ArrayList<>();
    final Map<Key, Mutation> seedResponses = new HashMap<>();
    final List<Motif> motifs;
    final int[][] coverage;
    final double[][] pairs;
    final Random proposals;
    final int budget;
    final double seconds;
    final long started = System.nanoTime();
    Row wt, best;
    int warmCount, attempts, step;
    double lastImprovement, secondsPerWork = 1e-7;
    String stop;

    EmSequenceSearch(EmEnergyModel model, long seed, int budget, double seconds) {
        this(model, Arrays.stream(model.states).map(Cache::new).toArray(Cache[]::new), new ArrayList<>(),
            new double[model.reference.length][model.reference.length], seed, budget, seconds);
    }
    EmSequenceSearch(EmSequenceSearch warm, long seed) {
        this(warm.model, Arrays.stream(warm.caches).map(Cache::new).toArray(Cache[]::new),
            warm.motifs, warm.pairs, seed, warm.budget, warm.seconds);
        rows.putAll(warm.rows);
        mutations.addAll(warm.mutations);
        seedResponses.putAll(warm.seedResponses);
        for (int i = 0; i < coverage.length; i++) coverage[i] = warm.coverage[i].clone();
        wt = warm.wt;
        best = warm.best;
        warmCount = rows.size();
    }
    private EmSequenceSearch(EmEnergyModel model, Cache[] caches, List<Motif> motifs, double[][] pairs, long seed, int budget, double seconds) {
        this.model = model;
        this.caches = caches;
        this.motifs = motifs;
        this.pairs = pairs;
        this.budget = budget;
        this.seconds = seconds;
        proposals = new Random(seed);
        coverage = Arrays.stream(model.alphabet).map(a -> new int[a.length]).toArray(int[][]::new);
    }

    double elapsed() { return (System.nanoTime() - started) / 1e9; }
    void check() {
        if (Thread.currentThread().isInterrupted()) throw new java.util.concurrent.CancellationException();
        if (warmCount > 0 && elapsed() >= seconds) throw new Horizon();
    }

    Row score(int[] seq) {
        Key key = new Key(seq);
        Row old = rows.get(key);
        if (old != null) return old;
        check();
        double[] z = new double[caches.length];
        for (int s = 0; s < z.length; s++) z[s] = caches[s].score(seq, this::check);
        double value = z[1] == Double.NEGATIVE_INFINITY || z[2] == Double.NEGATIVE_INFINITY
            ? Double.NaN : z[0] - z[1] - z[2];
        if (value == Double.POSITIVE_INFINITY) throw new ArithmeticException("Unrepresentable Em score");
        Row row = new Row(key.values, z, value);
        rows.put(key, row);
        if (Double.isFinite(value)) {
            for (int i = 0; i < seq.length; i++) coverage[i][seq[i]]++;
            if (best == null || value > best.score) { best = row; lastImprovement = elapsed(); }
        }
        return row;
    }

    // Pay for the WT and all one- and two-position substitutions once per design.
    void prepare() {
        int[] reference = model.reference;
        wt = score(reference);
        if (!Double.isFinite(wt.score)) throw new IllegalArgumentException("Sequence search requires a finite wild-type Em score");
        for (int i = 0; i < reference.length; i++) {
            for (int a = 0; a < model.alphabet[i].length; a++) if (a != reference[i]) {
                int[] seq = reference.clone();
                seq[i] = a;
                score(seq);
            }
        }
        for (int i = 0; i < reference.length; i++) for (int j = i + 1; j < reference.length; j++) {
            for (int a = 0; a < model.alphabet[i].length; a++) if (a != reference[i]) {
                for (int b = 0; b < model.alphabet[j].length; b++) if (b != reference[j]) {
                    int[] seq = reference.clone();
                    seq[i] = a;
                    seq[j] = b;
                    score(seq);
                }
            }
        }
        for (Row row : rows.values()) {
            if (row == wt || !Double.isFinite(row.score)) continue;
            int[] positions = changed(reference, row.seq);
            double synergy = 0;
            if (positions.length == 2) {
                int i = positions[0], j = positions[1];
                int[] first = reference.clone(), second = reference.clone();
                first[i] = row.seq[i];
                second[j] = row.seq[j];
                double x = rows.get(new Key(first)).score, y = rows.get(new Key(second)).score;
                if (Double.isFinite(x) && Double.isFinite(y)) synergy = row.score - x - y + wt.score;
                pairs[i][j] = pairs[j][i] = Math.max(pairs[i][j], Math.abs(synergy));
            }
            motifs.add(new Motif(positions, row, row.score - wt.score, synergy));
        }
        motifs.sort(Comparator.comparingDouble((Motif m) -> m.gain + .5 * Math.max(0, m.synergy)).reversed());
        Set<Key> blocks = new HashSet<>();
        for (Motif motif : motifs) if (blocks.add(new Key(motif.positions))) {
            Mutation m = capture(wt, motif.target, motif.positions);
            seedResponses.put(new Key(motif.target.seq), m);
            if (motif.positions.length == 2 && m.bounded()) {
                int i = motif.positions[0], j = motif.positions[1];
                pairs[i][j] = pairs[j][i] = pairs[i][j] + Math.min(10, m.width() / 10);
            }
        }
    }

    Mutation capture(Row source, Row target, int[] positions) {
        Region[] regions = new Region[caches.length];
        BitSet pinned = new BitSet();
        double[] low = new double[caches.length], high = new double[caches.length];
        for (int s = 0; s < caches.length; s++) {
            Cache cache = caches[s];
            Region r = regions[s] = cache.response(source.seq, target.seq, this::check);
            if (r == null) continue;
            r.variables().stream().forEach(p -> pinned.set(cache.state.global[p]));
            if (r.equals(cache.state.whole)) {
                low[s] = high[s] = target.logZ[s] - source.logZ[s];
                continue;
            }
            double[] a = cache.message(r, source.seq), b = cache.message(r, target.seq);
            low[s] = Double.POSITIVE_INFINITY;
            high[s] = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < a.length; j++) {
                if (a[j] == Double.NEGATIVE_INFINITY && b[j] == Double.NEGATIVE_INFINITY) continue;
                if (!Double.isFinite(a[j]) || !Double.isFinite(b[j])) {
                    low[s] = Double.NEGATIVE_INFINITY; high[s] = Double.POSITIVE_INFINITY; break;
                }
                low[s] = Math.min(low[s], b[j] - a[j]);
                high[s] = Math.max(high[s], b[j] - a[j]);
            }
        }
        Mutation m = new Mutation(source, positions, target, pinned, regions, low[0] - high[1] - high[2], high[0] - low[1] - low[2]);
        double gain = target.score - source.score;
        if (m.bounded() && (gain < m.lower - 1e-8 || gain > m.upper + 1e-8)) throw new ArithmeticException("Inconsistent Em response interval");
        if (mutations.size() == 512) mutations.remove(0);
        mutations.add(m);
        return m;
    }

    boolean available(Mutation m, int[] seq) {
        if (m == null) return false;
        for (int s = 0; s < caches.length; s++) {
            Region r = m.regions[s];
            if (r != null && !r.equals(caches[s].state.whole) && caches[s].message(r, seq) == null) return false;
        }
        return true;
    }

    List<Row> elites() {
        List<Row> ranked = rows.values().stream().filter(r -> Double.isFinite(r.score))
            .sorted(Comparator.comparingDouble(Row::score).reversed()).toList();
        List<Row> elites = new ArrayList<>(ranked.subList(0, Math.min(2, ranked.size())));
        if (ranked.size() > 2) {
            Row distant = Collections.max(ranked.subList(2, ranked.size()), Comparator.comparingInt(r ->
                Math.min(distance(r.seq, elites.get(0).seq), distance(r.seq, elites.get(1).seq))));
            elites.add(distant);
        }
        return elites;
    }

    void add(List<Action> out, Set<Key> seen, Kind kind, Row bg, int[] seq, double hint, double synergy, Mutation evidence, boolean batchOnly) {
        Key key = new Key(seq);
        if (rows.containsKey(key) || failed.contains(key) || !seen.add(key)) return;
        int[] positions = changed(bg.seq, seq);
        Action a = new Action(kind, bg, positions, List.of(key.values), hint, synergy);
        for (Mutation m : mutations) {
            if (!Arrays.equals(positions, m.positions) || !m.compatible(bg.seq)) continue;
            if (Arrays.stream(positions).allMatch(i -> seq[i] == m.target.seq[i]) && (a.match == null || m.width() < a.match.width())) a.match = m;
        }
        if (a.match != null && a.match.bounded() && bg.score + a.match.upper < best.score - 1e-8) return;
        a.evidence = evidence;
        a.batchOnly = batchOnly || available(a.match, seq);
        out.add(a);
    }

    List<Action> propose(List<Row> elites) {
        List<Action> out = new ArrayList<>();
        Set<Key> seen = new HashSet<>();
        int n = model.names.length;
        for (int k = 0; k < motifs.size() && out.size() < POOL / 2; k++) {
            Motif m = motifs.get(k < 12 ? k : (int)((step * 7L + k) % motifs.size()));
            for (Row bg : elites) {
                int[] seq = bg.seq.clone();
                for (int i : m.positions) seq[i] = m.target.seq[i];
                double discount = 1 + distance(model.reference, bg.seq) / (double)Math.max(1, n);
                add(out, seen, Kind.LOCAL, bg, seq, m.gain / discount, m.synergy, seedResponses.get(new Key(m.target.seq)), false);
                if (out.size() >= POOL / 2) break;
            }
        }
        int cap = Math.min(POOL - 8, out.size() + POOL / 4);
        for (int k = 0; k < motifs.size() && out.size() < cap; k++) {
            Motif m = motifs.get((int)((k + (long)step) % Math.min(40, motifs.size())));
            if (m.positions.length != 2) continue;
            List<Integer> extra = new ArrayList<>();
            for (int i = 0; i < n; i++) if (model.alphabet[i].length > 1 && i != m.positions[0] && i != m.positions[1]) extra.add(i);
            if (extra.isEmpty()) continue;
            extra.sort(Comparator.comparingDouble((Integer i) -> -(pairs[i][m.positions[0]] + pairs[i][m.positions[1]])));
            int i = extra.get((step / 8) % extra.size()), count = model.alphabet[i].length;
            Row bg = elites.get(k % elites.size());
            int[] seq = bg.seq.clone();
            for (int p : m.positions) seq[p] = m.target.seq[p];
            int aa = (int)((step + (long)k) % count);
            for (Motif single : motifs) if (single.positions.length == 1 && single.positions[0] == i) { aa = single.target.seq[i]; break; }
            seq[i] = seq[i] == aa ? (aa + 1 + step % (count - 1)) % count : aa;
            add(out, seen, Kind.EXPAND, bg, seq, m.gain, m.synergy, seedResponses.get(new Key(m.target.seq)), false);
        }
        for (Mutation m : mutations) {
            if (out.size() >= POOL - 6) break;
            for (Row bg : elites) if (m.compatible(bg.seq)) {
                int[] seq = bg.seq.clone();
                for (int i : m.positions) seq[i] = m.target.seq[i];
                add(out, seen, Kind.LOCAL, bg, seq, m.target.score - m.source.score, 0, null, true);
            }
        }
        for (int k = 0; k < POOL * 40 && out.size() < POOL - 4; k++) {
            if ((k & 63) == 0) check();
            Row bg = elites.get(proposals.nextInt(elites.size()));
            int[] seq = bg.seq.clone();
            for (int i = 0; i < n; i++) if (model.alphabet[i].length > 1) seq[i] = proposals.nextInt(model.alphabet[i].length);
            add(out, seen, Kind.REMOTE, bg, seq, 0, 0, null, false);
        }
        if (out.isEmpty()) {
            int[] seq = new int[n];
            for (int k = 0; k < 100_000 && out.isEmpty(); k++) {
                if ((k & 63) == 0) check();
                add(out, seen, Kind.REMOTE, elites.get(0), seq, 0, 0, null, false);
                int i = n - 1;
                while (i >= 0 && ++seq[i] >= model.alphabet[i].length) seq[i--] = 0;
                if (i < 0) break;
            }
        }
        record Group(Row background, Key positions) {}
        Map<Group, List<Action>> groups = new LinkedHashMap<>();
        for (Action a : out) groups.computeIfAbsent(new Group(a.background, new Key(a.positions)), g -> new ArrayList<>()).add(a);
        for (var group : groups.values()) if (group.size() >= 2 && out.size() < POOL - 1) {
            Action first = group.get(0);
            out.add(new Action(Kind.BATCH, first.background, first.positions,
                group.stream().limit(4).map(a -> a.candidates.get(0)).toList(),
                group.stream().mapToDouble(a -> a.hint).max().orElseThrow(), first.synergy));
        }
        out.removeIf(a -> a.batchOnly);
        return out;
    }

    double remaining() { return Math.max(0, seconds - elapsed()); }
    int remainingQueries() { return budget - attempts; }

    double novelty(Action a) {
        double max = 0;
        for (int[] seq : a.candidates) {
            double sum = 0;
            for (int i = 0; i < seq.length; i++) sum += 1 / Math.sqrt(1 + coverage[i][seq[i]]);
            max = Math.max(max, sum / Math.max(1, model.names.length));
        }
        return max;
    }

    static long reuse(Action a, List<Action> pool) {
        return pool.stream().filter(b -> a.background == b.background && Arrays.equals(a.positions, b.positions)).count();
    }

    static double information(Action a, long reuse, double novelty) {
        double info = .15 * Math.log1p(reuse) + novelty;
        if (a.kind == Kind.EXPAND) {
            if (a.evidence != null && a.evidence.bounded()) info += .3 * Math.log1p(a.evidence.width());
            info += .3 * Math.log1p(Math.abs(a.synergy));
        }
        if (a.kind == Kind.REMOTE) info++;
        return info;
    }

    record Pool(List<Action> actions, List<Row> elites) {}
    private static final Pool END = new Pool(List.of(), List.of());

    Pool pool() {
        if (attempts >= budget) { stop = "query_budget"; return END; }
        if ((long)rows.size() + failed.size() == model.sequenceCount) { stop = "exhausted"; return END; }
        check();
        List<Row> elites = elites();
        List<Action> actions = propose(elites);
        if (actions.isEmpty()) { stop = "no_candidates"; return END; }
        return new Pool(actions, elites);
    }

    record Execution(double seconds, double work) {}

    Execution execute(Action a, Consumer<Row> observe) {
        double begun = elapsed(), work = totalWork();
        Row captured = null;
        for (int[] seq : a.candidates) {
            if (attempts == budget) break;
            check();
            attempts++;
            Row row;
            try { row = score(seq); }
            catch (RootSelectionException | DPTableTooLargeException unavailable) {
                failed.add(new Key(seq));
                continue;
            }
            if (Double.isFinite(row.score)) {
                if (captured == null || row.score > captured.score) captured = row;
                if (observe != null) observe.accept(row);
            }
        }
        if (captured != null && a.positions.length <= 3) capture(a.background, captured, changed(a.background.seq, captured.seq));
        double duration = elapsed() - begun, executed = totalWork() - work;
        if (executed > 0) secondsPerWork = .8 * secondsPerWork + .2 * duration / executed;
        step++;
        return new Execution(duration, executed);
    }

    double totalWork() { return Arrays.stream(caches).mapToDouble(c -> c.work).sum(); }
    Map<String, Object> result(String route) {
        Map<String, String> sequence = new LinkedHashMap<>();
        for (int i = 0; i < best.seq.length; i++) sequence.put(model.names[i], model.alphabet[i][best.seq[i]]);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("route", route);
        out.put("sequence", sequence);
        out.put("em_log_k", best.score);
        out.put("em_log_z", Map.of("complex", best.logZ[0], "protein", best.logZ[1], "ligand", best.logZ[2]));
        out.put("warm_sequences", warmCount);
        out.put("search_queries", attempts);
        out.put("search_scores", rows.size() - warmCount);
        out.put("stop", stop);
        return out;
    }

    static void writeJson(Path path, Object value) throws IOException {
        Path dir = Files.createDirectories(path.toAbsolutePath().getParent());
        Path pending = Files.createTempFile(dir, ".sequence-", ".tmp");
        try {
            JSON.writerWithDefaultPrettyPrinter().writeValue(pending.toFile(), value);
            Files.move(pending, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(pending);
        }
    }
}
