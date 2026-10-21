package edu.duke.cs.osprey.branchdp;

import static org.junit.jupiter.api.Assertions.*;

import edu.duke.cs.osprey.astar.conf.RCs;
import java.lang.reflect.Field;
import java.util.LinkedHashSet;
import org.junit.jupiter.api.Test;

/**
 * Lightweight integration tests for the M-state mixed-radix indexing on a real
 * RootedTreeEdge: computeIndexInA(...) and its inverse decodeMStatePublic(...).
 *
 * Covers the remaining [TODO] unit tests in
 * BranchMARKStar_PAC_Reliability_Design.md:
 *   - computeIndexInA long mixed-radix tests
 *   - decodeMState(long) round-trip tests
 *
 * These exercise the long index path (> Integer.MAX_VALUE), which is the whole
 * point of the Phase B sharding work. We build a real edge (real RCs, real
 * constructor) and set only the M-position layout that the full-tree path would
 * otherwise compute, so we test the production index math without standing up a
 * conf space / DP run.
 */
public class TestMStateIndexing {

	// ---- helpers ----------------------------------------------------------

	/** RCs whose getNum(positions[i]) == cardinalities[i]; other positions have 1 RC. */
	private static RCs rcsFor(int[] positions, int[] cardinalities) {
		int maxPos = 0;
		for (int p : positions) {
			maxPos = Math.max(maxPos, p);
		}
		int[][] arr = new int[maxPos + 1][];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = new int[1];
		}
		for (int i = 0; i < positions.length; i++) {
			arr[positions[i]] = new int[cardinalities[i]];
		}
		return new RCs(arr);
	}

	/** Build a non-root edge and inject the sorted M positions the tree path would set. */
	private static RootedTreeEdge edgeFor(int[] positions, int[] cardinalities) {
		RCs rcs = rcsFor(positions, cardinalities);
		RootedTreeEdge edge = new RootedTreeEdge(null, null, new LinkedHashSet<>(), false, rcs);
		try {
			Field f = RootedTreeEdge.class.getDeclaredField("mPositionsSorted");
			f.setAccessible(true);
			f.set(edge, positions.clone());
		} catch (ReflectiveOperationException e) {
			throw new RuntimeException("failed to inject mPositionsSorted", e);
		}
		return edge;
	}

	/** Reference mixed-radix encode, most-significant position first (matches computeIndexInA). */
	private static long refIndex(int[] rcsIdx, int[] cardinalities) {
		long idx = 0;
		for (int i = 0; i < rcsIdx.length; i++) {
			idx = idx * cardinalities[i] + rcsIdx[i];
		}
		return idx;
	}

	/** Reference decode, inverse of refIndex. */
	private static int[] refDecode(long idx, int[] cardinalities) {
		int[] out = new int[cardinalities.length];
		long rem = idx;
		for (int i = cardinalities.length - 1; i >= 0; i--) {
			out[i] = (int) (rem % cardinalities[i]);
			rem /= cardinalities[i];
		}
		return out;
	}

	// ---- tests ------------------------------------------------------------

	@Test
	public void computeIndexInA_matchesReference_smallExhaustive() {
		int[] positions = {0, 1, 2};
		int[] cards = {2, 3, 4}; // 24 states, contiguous 0..23
		RootedTreeEdge edge = edgeFor(positions, cards);
		int total = 2 * 3 * 4;
		for (int flat = 0; flat < total; flat++) {
			int[] rcsIdx = refDecode(flat, cards);
			long idx = edge.computeIndexInA(rcsIdx);
			assertEquals(refIndex(rcsIdx, cards), idx, "vs reference, flat=" + flat);
			assertEquals(flat, idx, "index must be contiguous, flat=" + flat);
		}
	}

	@Test
	public void decodeMState_roundTrip_exhaustive() {
		int[] positions = {0, 1, 2, 3};
		int[] cards = {3, 5, 2, 7};
		RootedTreeEdge edge = edgeFor(positions, cards);
		long total = 3L * 5 * 2 * 7;
		for (long mIdx = 0; mIdx < total; mIdx++) {
			int[] decoded = edge.decodeMStatePublic(mIdx);
			assertArrayEquals(refDecode(mIdx, cards), decoded, "decode mismatch at mIdx=" + mIdx);
			assertEquals(mIdx, edge.computeIndexInA(decoded),
				"encode(decode(mIdx)) != mIdx at mIdx=" + mIdx);
		}
	}

	@Test
	public void nonContiguousPositions_useCorrectCardinalities() {
		// positions are not 0..k-1; cardinalities must be looked up per-position
		int[] positions = {2, 5, 9};
		int[] cards = {4, 3, 5}; // 60 states
		RootedTreeEdge edge = edgeFor(positions, cards);
		long total = 4L * 3 * 5;
		for (long mIdx = 0; mIdx < total; mIdx++) {
			int[] decoded = edge.decodeMStatePublic(mIdx);
			assertArrayEquals(refDecode(mIdx, cards), decoded, "decode at mIdx=" + mIdx);
			assertEquals(mIdx, edge.computeIndexInA(decoded), "round-trip at mIdx=" + mIdx);
		}
	}

	@Test
	public void computeIndexInA_exceedsIntRange_longCorrect() {
		// product = 1000 * 1000 * 1000 * 10 = 10_000_000_000 > Integer.MAX_VALUE
		int[] positions = {0, 1, 2, 3};
		int[] cards = {1000, 1000, 1000, 10};
		RootedTreeEdge edge = edgeFor(positions, cards);

		long last = edge.computeIndexInA(new int[]{999, 999, 999, 9});
		assertEquals(9_999_999_999L, last, "long mixed-radix overflowed into int range");
		assertTrue(last > Integer.MAX_VALUE, "test must exercise > Integer.MAX_VALUE");

		// representative interior index
		assertEquals(500L * 1000 * 1000 * 10,
			edge.computeIndexInA(new int[]{500, 0, 0, 0}));

		// just past the int boundary, encode/decode must be consistent
		long boundary = (long) Integer.MAX_VALUE + 1L; // 2_147_483_648
		int[] decoded = edge.decodeMStatePublic(boundary);
		assertEquals(boundary, edge.computeIndexInA(decoded),
			"round-trip failed across the Integer.MAX_VALUE boundary");
		assertArrayEquals(refDecode(boundary, cards), decoded);

		// and the very last state round-trips
		int[] lastDecoded = edge.decodeMStatePublic(9_999_999_999L);
		assertArrayEquals(new int[]{999, 999, 999, 9}, lastDecoded);
	}

	@Test
	public void rootEdgeIndexesToZero() {
		RCs rcs = rcsFor(new int[]{0, 1}, new int[]{3, 3});
		RootedTreeEdge rootEdge = new RootedTreeEdge(null, null, new LinkedHashSet<>(), true, rcs);
		assertEquals(0L, rootEdge.computeIndexInA(new int[]{2, 1}),
			"root edge must collapse to a single index 0");
	}
}
