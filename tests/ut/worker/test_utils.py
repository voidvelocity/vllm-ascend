import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.worker.utils import copy_snapshot_to_gpu


class TestQueryStartLocCopy(unittest.TestCase):
    def test_copy_delegates_to_buffer_copy_to_gpu(self):
        """The wrapper just forwards to ``CpuGpuBuffer.copy_to_gpu``.

        There is no snapshot/cached pinned buffer any more: ``buffer.cpu``
        is itself pinned, so a direct H2D from pinned memory is the
        safest path. Re-using a snapshot raced with the previous
        async H2D DMA and produced garbled query_start_loc values.
        """
        cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
        gpu = MagicMock()
        buf = MagicMock()
        buf.cpu = cpu
        buf.gpu = gpu
        buf.copy_to_gpu.return_value = "sentinel"

        result = copy_snapshot_to_gpu(buf)
        buf.copy_to_gpu.assert_called_once_with()
        self.assertEqual(result, "sentinel")
