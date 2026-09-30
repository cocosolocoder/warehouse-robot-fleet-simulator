import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


class PathfindingTests(unittest.TestCase):
    def test_shortest_path_avoids_obstacle(self) -> None:
        grid = GridMap(3, 3, frozenset({(1, 0)}))
        path = shortest_path(grid, (0, 0), (2, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertNotIn((1, 0), path)
        self.assertEqual(len(path), 4)


class SimulatorTests(unittest.TestCase):
    def test_task_completes_and_metrics_are_recorded(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 2),
            [Robot("R-01", (0, 0))],
            [Task("T-01", (1, 0), (3, 0))],
        )
        for _ in range(3):
            simulator.step()
        self.assertTrue(simulator.tasks["T-01"].completed)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.metrics()["distance_total"], 3)


if __name__ == "__main__":
    unittest.main()

