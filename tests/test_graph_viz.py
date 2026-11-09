import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import param
from bokeh.core.serialization import Serializer
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

import GDLModules.graph_viz as graph_viz


class GraphExpansionTests(unittest.TestCase):
    def setUp(self):
        # Radius graph is a three-node chain plus an isolated point.
        # Symmetric 1-NN also connects the last point to the chain.
        self.points = np.array(
            [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [1, 0, 0]],
            dtype=np.float32,
        )
        self.view = graph_viz.GraphExplorer(self.points, radius=0.11, k=1)

    def test_expand_and_rewind_reuse_geometry_and_traversals(self):
        with patch.object(graph_viz, "dijkstra", wraps=graph_viz.dijkstra) as traverse:
            self.view.select_seed(0)
            self.assertEqual(traverse.call_count, 2)
            with patch.object(graph_viz, "_edge_coordinates", wraps=graph_viz._edge_coordinates) as geometry:
                for _ in range(3):
                    self.view._on_expand(None)
                self.assertEqual(self.view.distances["radius"].tolist(), [0, 1, 2, -1])
                self.assertEqual(self.view.distances["knn"].tolist(), [0, 1, 2, 3])
                self.view._set_depth(1)
                self.assertEqual(self.view.distances["radius"].tolist(), [0, 1, -1, -1])
                self.view._set_depth(3)
                self.assertEqual(traverse.call_count, 2)
                geometry.assert_not_called()

    def test_one_plot_update_per_expansion_without_background_traces(self):
        self.view.select_seed(0)
        with patch.object(self.view.figure, "_send_update_msg", wraps=self.view.figure._send_update_msg) as send:
            self.view._on_expand(None)
            self.assertEqual(send.call_count, 1)
            trace_ids = send.call_args.kwargs["trace_indexes"]
            self.assertTrue(set(trace_ids).issubset({2, 3, 4, 7, 8, 9}))

    def test_batched_updates_serialize_for_browser_transport(self):
        self.view.select_seed(0)
        with patch.object(self.view, "_send_plot_update") as send:
            self.view._on_expand(None)
            self.assertEqual(send.call_count, 1)
            Serializer().encode(send.call_args.kwargs)
            send.reset_mock()
            self.view.radius.value = 0.05
            self.view._rebuild("radius")
            Serializer().encode(send.call_args.kwargs)

    def test_slider_only_updates_after_release(self):
        self.view.select_seed(0)
        self.view.depth.value = 2
        self.assertEqual(self.view.distances["radius"].tolist(), [0, -1, -1, -1])
        with param.edit_constant(self.view.depth):
            self.view.depth.value_throttled = 2
        self.assertEqual(self.view.distances["radius"].tolist(), [0, 1, 2, -1])

    def test_graph_change_only_invalidates_its_own_cache(self):
        self.view.select_seed(0)
        self.view._set_depth(2)
        knn_cache = self.view._expansion_cache["knn"]
        with patch.object(graph_viz, "dijkstra", wraps=graph_viz.dijkstra) as traverse:
            self.view.radius.value = 0.05
            self.view._rebuild("radius")
            self.assertEqual(traverse.call_count, 1)
        self.assertIs(self.view._expansion_cache["knn"], knn_cache)
        self.assertEqual(self.view.distances["radius"].tolist(), [0, -1, -1, -1])

    def test_camera_events_do_not_compute_graph_geometry_or_traversal(self):
        self.view.select_seed(0)
        with patch.object(graph_viz, "dijkstra") as traverse, patch.object(graph_viz, "_edge_coordinates") as geometry:
            self.view.plot.relayout_data = {"scene.camera.eye": {"x": 2, "y": 1, "z": 1}}
            traverse.assert_not_called()
            geometry.assert_not_called()

    def test_highlighted_nodes_can_be_selected_and_cleared(self):
        self.view.select_seed(0)
        self.view._set_depth(2)
        self.view._on_click(SimpleNamespace(new={"points": [{"curveNumber": 3, "pointNumber": 0}]}))
        self.assertEqual(self.view.seed, 1)
        self.assertEqual(self.view.depth.value, 0)
        self.view._on_clear(None)
        self.assertIsNone(self.view.seed)
        self.assertTrue(self.view.expand.disabled)
        for index in (2, 3, 4, 7, 8, 9):
            self.assertEqual(len(self.view.figure.data[index].x), 0)

    def test_radius_search_buffers_result_and_preserves_selection(self):
        self.view.select_seed(0)
        self.view._set_depth(2)
        other_graph = self.view.graphs["knn"]
        with patch.object(graph_viz, "_edge_coordinates", wraps=graph_viz._edge_coordinates) as geometry:
            asyncio.run(self.view._run_search("radius"))
            self.assertEqual(geometry.call_count, 1)
        self.assertFalse(self.view.error.visible, self.view.error.object)
        result = self.view.search_results["radius"]
        self.assertAlmostEqual(self.view.radius.value, result.value * 1.1)
        self.assertEqual(self.view.radius.value, self.view.radius.value_throttled)
        self.assertGreaterEqual(self.view.radius.end, self.view.radius.value)
        self.assertEqual(connected_components(self.view.graphs["radius"], directed=False, return_labels=False), 1)
        self.assertIs(self.view.graphs["knn"], other_graph)
        self.assertEqual(self.view.seed, 0)
        self.assertEqual(self.view.depth.value, 2)
        self.assertIn("Connectivity threshold", self.view.radius_search_status.object)
        self.assertFalse(self.view.search_radius.disabled)

    def test_k_search_expands_slider_range(self):
        x = np.concatenate((np.linspace(0, 0.05, 51), np.linspace(10, 10.05, 51)))
        points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x))).astype(np.float32)
        view = graph_viz.GraphExplorer(points, radius=0.01, k=1)
        self.assertEqual(view.k.end, 50)
        asyncio.run(view._run_search("knn"))
        self.assertFalse(view.error.visible, view.error.object)
        self.assertEqual(view.k.value, 51)
        self.assertEqual(view.k.value_throttled, 51)
        self.assertGreaterEqual(view.k.end, 51)
        self.assertIn("Minimum connected k", view.k_search_status.object)

    def test_failed_search_restores_controls_without_changing_graph(self):
        previous = self.view.graphs["radius"]
        with patch.object(self.view, "_search", side_effect=RuntimeError("test failure")):
            asyncio.run(self.view._run_search("radius"))
        self.assertTrue(self.view.error.visible)
        self.assertIn("test failure", self.view.error.object)
        self.assertIs(self.view.graphs["radius"], previous)
        self.assertFalse(self.view._search_running)
        self.assertFalse(self.view.search_radius.disabled)
        self.assertFalse(self.view.search_radius.loading)


class ConnectivitySearchTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [1, 0, 0]], dtype=np.float32)
        self.index = NearestNeighbors(algorithm="kd_tree", n_jobs=1).fit(self.points)

    def test_radius_search_returns_a_connected_upper_bound(self):
        for initial in (0.005, 2.0):
            with self.subTest(initial=initial):
                result = graph_viz.find_connected_radius(self.index, initial, tolerance=1e-5)
                self.assertLessEqual(result.value - result.lower_bound, 1e-5)
                self.assertAlmostEqual(result.value, 0.8, delta=1e-5)
                self.assertEqual(connected_components(result.graph, directed=False, return_labels=False), 1)
                below = graph_viz._neighbor_graph(self.index, "radius", result.lower_bound)
                self.assertGreater(connected_components(below, directed=False, return_labels=False), 1)
                self.assertLess(result.evaluations, 40)

    def test_radius_zero_for_identical_points(self):
        index = NearestNeighbors(algorithm="kd_tree").fit(np.zeros((3, 3)))
        result = graph_viz.find_connected_radius(index)
        self.assertEqual(result.value, 0)
        self.assertEqual(result.evaluations, 1)

    def test_invalid_radius_search_settings(self):
        for tolerance in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                graph_viz.find_connected_radius(self.index, tolerance=tolerance)

    def test_k_search_finds_exact_minimum_for_two_clusters(self):
        points = np.array([[x, 0, 0] for x in (0, 0.1, 0.2, 10, 10.1, 10.2)])
        index = NearestNeighbors(algorithm="kd_tree").fit(points)
        for initial in (1, 5):
            with self.subTest(initial=initial):
                result = graph_viz.find_connected_k(index, initial)
                self.assertEqual(result.value, 3)
                self.assertEqual(result.lower_bound, 2)
                self.assertEqual(connected_components(result.graph, directed=False, return_labels=False), 1)
                below = graph_viz._neighbor_graph(index, "knn", 2)
                self.assertEqual(connected_components(below, directed=False, return_labels=False), 2)

    def test_k_already_connected_at_one(self):
        result = graph_viz.find_connected_k(self.index)
        self.assertEqual(result.value, 1)
        self.assertEqual(result.evaluations, 1)


if __name__ == "__main__":
    unittest.main()
