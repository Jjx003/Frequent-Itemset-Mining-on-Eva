from typing import List

import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


class DivExplorer(AbstractFunction):
    @setup(cacheable=True, function_type="FeatureExtraction", batchable=False)
    def setup(
        self,
        min_support: float = 0.1,
        max_len: int = 3,
        metric: str = "fpr",
        ignore_cols: List[str] = None,
    ):
        self.min_support = min_support
        self.max_len = max_len
        self.metric = metric
        self.ignore_cols = ignore_cols
        pass

    @property
    def name(self) -> str:
        return "DivExplorer"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["predicted", "true"],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["itemset", "support"],
                column_types=[NdArrayType.STR, NdArrayType.FLOAT32],
            )
        ],
    )
    def forward(self):
        pass
