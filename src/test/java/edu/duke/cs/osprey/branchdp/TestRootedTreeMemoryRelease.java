package edu.duke.cs.osprey.branchdp;

import edu.duke.cs.osprey.astar.conf.RCs;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Array;
import java.util.LinkedHashSet;

import static org.junit.jupiter.api.Assertions.*;

/** Deterministic cleanup for the hundred-GiB rooted DP working set. */
public class TestRootedTreeMemoryRelease {

    private static void set(RootedTreeEdge edge, String name, Object value) {
        try {
            Field field = RootedTreeEdge.class.getDeclaredField(name);
            field.setAccessible(true);
            field.set(edge, value);
        } catch (ReflectiveOperationException ex) {
            throw new RuntimeException(ex);
        }
    }

    private static Object get(RootedTreeEdge edge, String name) {
        try {
            Field field = RootedTreeEdge.class.getDeclaredField(name);
            field.setAccessible(true);
            return field.get(edge);
        } catch (ReflectiveOperationException ex) {
            throw new RuntimeException(ex);
        }
    }

    private static Object emptyArrayForField(String name) {
        try {
            Field field = RootedTreeEdge.class.getDeclaredField(name);
            return Array.newInstance(field.getType().getComponentType(), 0);
        } catch (ReflectiveOperationException ex) {
            throw new RuntimeException(ex);
        }
    }

    private static RootedTreeEdge populatedEdge(RootedTreeNode parent,
                                                RootedTreeNode child,
                                                RCs rcs) {
        RootedTreeEdge edge = new RootedTreeEdge(parent, child,
                new LinkedHashSet<>(), false, rcs);
        set(edge, "dpTable", new ShardedDPTable(17L, 4));
        set(edge, "enumeratedCount", new int[17]);
        set(edge, "sortedLambdaIndices", new int[17][3]);
        set(edge, "lambdaOnlyRigid", new double[3]);
        set(edge, "lambdaOnlyMin", new double[3]);
        set(edge, "fullEnergyRigid", new double[17][3]);
        set(edge, "fullEnergyMin", new double[17][3]);
        set(edge, "childFoldPlans", emptyArrayForField("childFoldPlans"));
        set(edge, "Fset", new LinkedHashSet<RootedTreeEdge>());
        return edge;
    }

    private static void assertLargeStorageReleased(RootedTreeEdge edge) {
        assertFalse(edge.hasDPTable());
        assertNull(edge.getFset());
        assertNull(get(edge, "enumeratedCount"));
        assertNull(get(edge, "sortedLambdaIndices"));
        assertNull(get(edge, "lambdaOnlyRigid"));
        assertNull(get(edge, "lambdaOnlyMin"));
        assertNull(get(edge, "fullEnergyRigid"));
        assertNull(get(edge, "fullEnergyMin"));
        assertNull(get(edge, "childFoldPlans"));
    }

    @Test
    public void postOrderReleaseClearsEveryEdgeAndIsIdempotent() {
        RCs rcs = new RCs(new int[][]{{0, 1}});
        RootedTreeNode root = new RootedTreeNode(-3, false, -1, -1);
        RootedTreeNode left = new RootedTreeNode(0, true, 0, 0);
        RootedTreeNode right = new RootedTreeNode(1, true, 0, 0);
        root.setLeftChild(left);
        root.setRightChild(right);
        left.setParent(root);
        right.setParent(root);

        RootedTreeEdge leftEdge = populatedEdge(root, left, rcs);
        RootedTreeEdge rightEdge = populatedEdge(root, right, rcs);
        left.setChildOfEdge(leftEdge);
        right.setChildOfEdge(rightEdge);

        @SuppressWarnings("unchecked")
        LinkedHashSet<RootedTreeEdge> leftFset =
                (LinkedHashSet<RootedTreeEdge>) get(leftEdge, "Fset");
        leftFset.add(rightEdge);

        RootedTreeEdge.postOrderReleaseLargeMemory(root);
        assertLargeStorageReleased(leftEdge);
        assertLargeStorageReleased(rightEdge);

        assertDoesNotThrow(() -> RootedTreeEdge.postOrderReleaseLargeMemory(root));
        assertLargeStorageReleased(leftEdge);
        assertLargeStorageReleased(rightEdge);
    }
}
