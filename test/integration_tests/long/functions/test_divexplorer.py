import unittest

import pandas as pd

compas_dataset = '/home/jeff/evadb/data/divexplorer/compas_discretized.csv'

class DivExplorerTest(unittest.TestCase):
    
    def test_divexplorer_runs(self):
        from evadb.functions.divexplorer import DivExplorer

        divex = DivExplorer(
            min_support=0.1,
            ignore_cols=None,
            th_redundancy=None,
            top_k=None,
        )

        df = pd.read_csv(compas_dataset)
        result = divex(df, 'predicted', 'class')
        print(result)

