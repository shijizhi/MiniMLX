import os
import subprocess
import sys
import time
import unittest


class SupervisorLifecycleTests(unittest.TestCase):
    def test_runtime_exits_when_its_mini_mlx_parent_disappears(self):
        bootstrap = r"""
import os
import subprocess
import sys
environment = os.environ.copy()
environment["MINIMLX_SUPERVISOR_PID"] = str(os.getpid())
child = subprocess.Popen([
    sys.executable,
    "-c",
    "from minimlx_runtime.supervisor import install_parent_watchdog; "
    "import time; install_parent_watchdog(); time.sleep(30)",
], env=environment)
print(child.pid, flush=True)
"""
        runner = subprocess.Popen(
            [sys.executable, "-c", bootstrap],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertIsNotNone(runner.stdout)
        child_pid = int(runner.stdout.readline().strip())
        runner.stdout.close()
        runner.wait(timeout=5)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("supervised Python process survived its parent")


if __name__ == "__main__":
    unittest.main()
