import time
from typing import List, Optional

from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer.FP_Divergence import FP_Divergence
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


class DivExplorer(AbstractFunction):
    @setup(cacheable=True, function_type="FeatureExtraction", batchable=True)
    def setup(
        self,
        min_support: float = 0.1,
        max_len: int = 3,
        # metrics: List[str] = "fpr",
        metric: str = "d_fpr",
        ignore_cols: List[str] = None,
        th_redundancy: Optional[float] = None,
        top_k: Optional[int] = None,
        discretize: bool = False,
        discretize_bins: int = 5,
    ):
        self.min_support = min_support
        self.max_len = max_len
        self.metric = metric
        self.ignore_cols = ignore_cols
        self.th_redundancy = th_redundancy
        self.top_k = top_k
        self.discretize = discretize
        self.discretize_bins = discretize_bins

    @property
    def name(self) -> str:
        return "DivExplorer"

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Discretize the dataframe using pandas.qcut

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to discretize. Ignores the 'class' and 'predicted' columns.
        '''
        for col in df.columns:
            if col not in ['class', 'predicted']:
                continue
            df[col] = pd.qcut(df[col], self.discretize_bins, labels=False,
                              duplicates='drop')

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["class", "predicted"],
                column_types=[
                    NdArrayType.UINT8,
                    NdArrayType.UINT8,
                ],
                column_shapes=[(None,), (None,)]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["itemsets", "support", "support_count", "fpr", "d_fpr", "t_value_fp"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.UINT8,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,), (None,), (None,), (None,)]
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Runs DivExplorer on the given dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to run DivExplorer on. Must have 'class' and 'predicted' columns.
        '''
        if self.discretize:
            df = self.discretize(df)

        assert 'class' in df.columns, "Must have 'class' column"
        assert 'predicted' in df.columns, "Must have 'predicted' column"

        t1 = time.time()
        fp_diver = FP_DivergenceExplorer(
            df, true_class_name="class", 
            predicted_class_name="predicted", class_map={'N': 0, 'P': 1}
        )
        print('a', time.time() - t1)
        result_divexplore = fp_diver.getFrequentPatternDivergence(
            min_support=self.min_support, metrics=[self.metric], # TODO: Add max_len to DivExplorer, and multiple metrics
        )
        t1 = time.time()
        if self.top_k is not None:
            fp_divergence_metric = FP_Divergence(
                result_divexplore, self.metric
            )
            print('b', time.time() - t1)
            t1 = time.time()
            topK_df_metric = fp_divergence_metric.getDivergenceTopKDf(
                K=self.top_k, th_redundancy=self.th_redundancy
            )
            print('c', time.time() - t1)
            return topK_df_metric

        return result_divexplore


        # return df
