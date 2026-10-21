package edu.duke.cs.osprey.branchdp;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Correctness tests for the sharded final-DP table abstraction.
 *
 * This covers the lynchpin of the Phase B sharding work in
 * BranchMARKStar_PAC_Reliability_Design.md: dense-vs-sharded value equivalence
 * and the long shard-routing math, which must hold exactly so the sharded path
 * cannot silently return wrong DP values on large cases.
 *
 * These are pure unit tests over the DPTable implementations: no conf space /
 * protein setup required. (The edge-level mixed-radix index math in
 * RootedTreeEdge.computeIndexInA is an instance method tied to a constructed
 * edge; it is left to an integration-level test.)
 */
public class TestDPTable {

	@TempDir
	Path tempDir;

	// deterministic, distinct lower/upper values per index
	private static double lowerVal(long idx) { return -1.5 * (idx + 1); }
	private static double upperVal(long idx) { return  3.25 * (idx + 1) + 0.5; }

	// ---- dense vs sharded equivalence -------------------------------------

	private void checkDenseShardedEquivalent(int size, int shardSize) {
		DPTable dense = new DenseDPTable(size);
		DPTable sharded = new ShardedDPTable(size, shardSize);

		assertEquals(size, dense.size());
		assertEquals(size, sharded.size());
		assertTrue(dense.isDenseArrayBacked());
		assertFalse(sharded.isDenseArrayBacked());

		// write through both, in several strides to stress the routing math
		for (int step : new int[] { 1, 3, 7 }) {
			for (long idx = 0; idx < size; idx += step) {
				dense.set(idx, lowerVal(idx), upperVal(idx));
				sharded.set(idx, lowerVal(idx), upperVal(idx));
			}
		}

		for (long idx = 0; idx < size; idx++) {
			String where = " (size=" + size + ", shardSize=" + shardSize + ", idx=" + idx + ")";
			assertEquals(lowerVal(idx), dense.lower(idx), 0.0, "dense lower" + where);
			assertEquals(upperVal(idx), dense.upper(idx), 0.0, "dense upper" + where);
			assertEquals(dense.lower(idx), sharded.lower(idx), 0.0, "sharded vs dense lower" + where);
			assertEquals(dense.upper(idx), sharded.upper(idx), 0.0, "sharded vs dense upper" + where);
		}
	}

	@Test
	public void denseShardedEquivalence_variousShapes() {
		// exact multiples, non-multiples, size==1, and shardSize >= size (single shard)
		int[][] cases = {
			{1, 1}, {1, 8}, {7, 1}, {8, 4}, {8, 3}, {100, 16},
			{1000, 7}, {1003, 7}, {1024, 1024}, {1025, 1024}, {5000, 97},
		};
		for (int[] c : cases) {
			checkDenseShardedEquivalent(c[0], c[1]);
		}
	}

	@Test
	public void shardBoundaryRouting() {
		// 13 states with shardSize 4 -> shards of (4,4,4,1); check straddling indices
		int shardSize = 4;
		int size = 13;
		ShardedDPTable t = new ShardedDPTable(size, shardSize);
		for (long idx = 0; idx < size; idx++) {
			t.set(idx, lowerVal(idx), upperVal(idx));
		}
		for (long idx : new long[] {0, 3, 4, 7, 8, 11, 12}) {
			assertEquals(lowerVal(idx), t.lower(idx), 0.0, "lower at boundary " + idx);
			assertEquals(upperVal(idx), t.upper(idx), 0.0, "upper at boundary " + idx);
		}
	}

	@Test
	public void manyShardsRoutingIsContiguous() {
		// shardSize 1 forces one shard per index: every routing decision exercised
		int size = 257;
		ShardedDPTable t = new ShardedDPTable(size, 1);
		for (long idx = 0; idx < size; idx++) {
			t.set(idx, lowerVal(idx), upperVal(idx));
		}
		for (long idx = 0; idx < size; idx++) {
			assertEquals(lowerVal(idx), t.lower(idx), 0.0, "lower at idx " + idx);
			assertEquals(upperVal(idx), t.upper(idx), 0.0, "upper at idx " + idx);
		}
	}

	@Test
	public void fillSetsAllEntries_bothBackings() {
		DPTable dense = new DenseDPTable(50);
		DPTable sharded = new ShardedDPTable(50, 7);
		dense.fill(-9.0, 9.0);
		sharded.fill(-9.0, 9.0);
		for (long idx = 0; idx < 50; idx++) {
			assertEquals(-9.0, dense.lower(idx), 0.0);
			assertEquals(9.0, dense.upper(idx), 0.0);
			assertEquals(-9.0, sharded.lower(idx), 0.0);
			assertEquals(9.0, sharded.upper(idx), 0.0);
		}
	}

	@Test
	public void outOfBoundsRejected_sharded() {
		ShardedDPTable t = new ShardedDPTable(10, 4);
		assertThrows(IndexOutOfBoundsException.class, () -> t.lower(10));
		assertThrows(IndexOutOfBoundsException.class, () -> t.lower(-1));
		assertThrows(IndexOutOfBoundsException.class, () -> t.set(10, 0, 0));
	}

	@Test
	public void estimatedBytesAndFlags() {
		assertEquals(2L * 10 * 8L, new DenseDPTable(10).estimatedBytes());
		assertEquals(2L * 10 * 8L, new ShardedDPTable(10, 4).estimatedBytes());
	}

	@Test
	public void denseRejectsOverIntLimit() {
		assertThrows(IllegalArgumentException.class,
			() -> new DenseDPTable((long) Integer.MAX_VALUE + 1L));
	}

	@Test
	public void shardedUnsafeArrayAccessThrows() {
		ShardedDPTable t = new ShardedDPTable(10, 4);
		assertThrows(IllegalStateException.class, t::lowerArrayUnsafe);
		assertThrows(IllegalStateException.class, t::upperArrayUnsafe);
	}

	@Test
	public void denseUnsafeArrayMatchesAccessors() {
		DenseDPTable t = new DenseDPTable(8);
		for (long idx = 0; idx < 8; idx++) {
			t.set(idx, lowerVal(idx), upperVal(idx));
		}
		double[] lo = t.lowerArrayUnsafe();
		double[] hi = t.upperArrayUnsafe();
		for (int i = 0; i < 8; i++) {
			assertEquals(t.lower(i), lo[i], 0.0);
			assertEquals(t.upper(i), hi[i], 0.0);
		}
	}

	@Test
	public void indexedCopyWritesScatter_denseAndSharded() {
		for (DPTable t : new DPTable[] { new DenseDPTable(12), new ShardedDPTable(12, 5) }) {
			t.fill(Double.NaN, Double.NaN);
			long[] idx = { 7, 0, 11, 4 };
			double[] lo = { 1.0, 2.0, 3.0, 4.0 };
			double[] hi = { 11.0, 12.0, 13.0, 14.0 };
			t.copyFromIndexed(idx, DoubleBuffer.wrap(lo), DoubleBuffer.wrap(hi), idx.length);
			for (int i = 0; i < idx.length; i++) {
				assertEquals(lo[i], t.lower(idx[i]), 0.0);
				assertEquals(hi[i], t.upper(idx[i]), 0.0);
			}
			assertTrue(Double.isNaN(t.lower(1)));
			assertTrue(Double.isNaN(t.upper(1)));
		}
	}

	@Test
	public void readPairReadsBothBackingsAndShardBoundaries() {
		for (DPTable t : new DPTable[] { new DenseDPTable(13), new ShardedDPTable(13, 4) }) {
			for (long idx = 0; idx < t.size(); idx++) {
				t.set(idx, lowerVal(idx), upperVal(idx));
			}
				double[] lo = new double[4];
				double[] hi = new double[4];
				DoubleBuffer directLo = DoubleBuffer.allocate(4);
				DoubleBuffer directHi = DoubleBuffer.allocate(4);
				long[] indices = { 0, 3, 4, 12 };
				for (int i = 0; i < indices.length; i++) {
					t.readPair(indices[i], lo, hi, i);
					t.readPair(indices[i], directLo, directHi, i);
					assertEquals(lowerVal(indices[i]), lo[i], 0.0);
					assertEquals(upperVal(indices[i]), hi[i], 0.0);
					assertEquals(lowerVal(indices[i]), directLo.get(i), 0.0);
					assertEquals(upperVal(indices[i]), directHi.get(i), 0.0);
				}
				assertThrows(IndexOutOfBoundsException.class,
					() -> t.readPair(t.size(), lo, hi, 0));
				assertThrows(IndexOutOfBoundsException.class,
					() -> t.readPair(t.size(), directLo, directHi, 0));
		}
	}

	@Test
	public void mappedTableCrossChunkOperationsAndCleanup() {
		MappedDPTable t = new MappedDPTable(13, 4, tempDir);
		Path path = t.pathForTesting();
		assertTrue(Files.exists(path));
		assertEquals(13L, t.size());
		assertEquals(2L * 13L * Double.BYTES, t.estimatedBytes());
		assertFalse(t.isDenseArrayBacked());
		assertFalse(t.supportsArrayChunks());
		assertTrue(t.isFileBacked());
		assertThrows(IllegalStateException.class, t::lowerArrayUnsafe);
		assertThrows(IllegalStateException.class, t::upperArrayUnsafe);
		assertThrows(IllegalStateException.class, t::lowerChunks);
		assertThrows(IllegalStateException.class, t::upperChunks);

		t.fill(Double.NaN, Double.NaN);
		for (long idx : new long[] { 0, 3, 4, 7, 8, 11, 12 }) {
			t.set(idx, lowerVal(idx), upperVal(idx));
			assertEquals(lowerVal(idx), t.lower(idx), 0.0);
			assertEquals(upperVal(idx), t.upper(idx), 0.0);
		}

		double[] copyLo = new double[9];
		double[] copyHi = new double[9];
		for (int i = 0; i < copyLo.length; i++) {
			copyLo[i] = 100.0 + i;
			copyHi[i] = 200.0 + i;
		}
		t.copyFrom(2L, direct(copyLo), direct(copyHi), copyLo.length);
		for (int i = 0; i < copyLo.length; i++) {
			assertEquals(copyLo[i], t.lower(2L + i), 0.0);
			assertEquals(copyHi[i], t.upper(2L + i), 0.0);
		}

		long[] scatter = { 12, 0, 7, 4 };
		double[] scatterLo = { -12.0, -10.0, -7.0, -4.0 };
		double[] scatterHi = { 12.0, 10.0, 7.0, 4.0 };
		t.copyFromIndexed(scatter, direct(scatterLo), direct(scatterHi), scatter.length);
		double[] pairLo = new double[scatter.length];
		double[] pairHi = new double[scatter.length];
		DoubleBuffer directLo = direct(new double[scatter.length]);
		DoubleBuffer directHi = direct(new double[scatter.length]);
		for (int i = 0; i < scatter.length; i++) {
			t.readPair(scatter[i], pairLo, pairHi, i);
			t.readPair(scatter[i], directLo, directHi, i);
			assertEquals(scatterLo[i], pairLo[i], 0.0);
			assertEquals(scatterHi[i], pairHi[i], 0.0);
			assertEquals(scatterLo[i], directLo.get(i), 0.0);
			assertEquals(scatterHi[i], directHi.get(i), 0.0);
		}

		assertThrows(IndexOutOfBoundsException.class, () -> t.lower(-1));
		assertThrows(IndexOutOfBoundsException.class, () -> t.upper(13));
		assertThrows(IndexOutOfBoundsException.class,
				() -> t.copyFrom(10, direct(new double[4]), direct(new double[4]), 4));
		assertThrows(IndexOutOfBoundsException.class,
				() -> t.copyFromIndexed(new long[] { 13 }, direct(new double[1]),
						direct(new double[1]), 1));

		t.close();
		t.close();
		assertFalse(Files.exists(path));
		assertThrows(IllegalStateException.class, () -> t.lower(0));
	}

	@Test
	public void mappedModeThresholdSelectionUsesLongStateCounts() {
		String oldMode = System.getProperty("branchdp.dp.tableMode");
		String oldThreshold = System.getProperty("branchdp.dp.mmap.thresholdBytes");
		try {
			System.setProperty("branchdp.dp.tableMode", "auto_mmap");
			System.setProperty("branchdp.dp.mmap.thresholdBytes", "160");
			assertFalse(RootedTreeEdge.shouldUseFileBackedDPTable(9));
			assertTrue(RootedTreeEdge.shouldUseFileBackedDPTable(10));
			assertTrue(RootedTreeEdge.shouldUseFileBackedDPTable(
					(long) Integer.MAX_VALUE + 1L));
			System.setProperty("branchdp.dp.tableMode", "auto");
			assertFalse(RootedTreeEdge.shouldUseFileBackedDPTable(Long.MAX_VALUE / 32L));
			System.setProperty("branchdp.dp.tableMode", "mmap");
			assertTrue(RootedTreeEdge.shouldUseFileBackedDPTable(1));
		} finally {
			restoreProperty("branchdp.dp.tableMode", oldMode);
			restoreProperty("branchdp.dp.mmap.thresholdBytes", oldThreshold);
		}
	}

	private static DoubleBuffer direct(double[] values) {
		DoubleBuffer out = ByteBuffer.allocateDirect(values.length * Double.BYTES)
				.order(ByteOrder.nativeOrder()).asDoubleBuffer();
		out.put(values);
		out.rewind();
		return out;
	}

	private static void restoreProperty(String key, String value) {
		if (value == null) System.clearProperty(key);
		else System.setProperty(key, value);
	}
}
