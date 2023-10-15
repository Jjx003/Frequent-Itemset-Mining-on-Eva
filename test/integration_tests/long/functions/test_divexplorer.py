import unittest

import pandas as pd

from evadb.server.command_handler import execute_query_fetch_all
from evadb.configuration.constants import EvaDB_DATABASE_DIR, EvaDB_ROOT_DIR
from test.util import (
    get_evadb_for_testing,
    shutdown_ray,
    load_functions_for_testing
)

compas_dataset = '/home/jeff/evadb/data/divexplorer/compas_discretized.csv'

class DivExplorerTest(unittest.TestCase):
    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        load_functions_for_testing(self.evadb, mode="debug")

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, "DROP TABLE IF EXISTS MyCompas;")
    
    # def test_divexplorer_runs(self):
    #     from evadb.functions.divexplorer import DivExplorer

    #     divex = DivExplorer(
    #         min_support=0.1,
    #         ignore_cols=None,
    #         th_redundancy=None,
    #         top_k=None,
    #     )

    #     df = pd.read_csv(compas_dataset)
    #     result = divex(df)
    #     self.assertTrue(result is not None)

    # def test_divexplorer_runs_with_top_k(self):
    #     from evadb.functions.divexplorer import DivExplorer

    #     divex = DivExplorer(
    #         min_support=0.05,
    #         ignore_cols=None,
    #         th_redundancy=0.05,
    #         top_k=10,
    #     )

    #     df = pd.read_csv(compas_dataset)
    #     result = divex(df)
    #     self.assertTrue(result is not None)

    def test_divexplorer_via_query(self):
        from evadb.functions.divexplorer import DivExplorer

        create_table_query = """
            CREATE TABLE IF NOT EXISTS MyCompas (
                age INTEGER,
                charge TEXT(30),
                race TEXT(30),
                sex TEXT(10),
                n_prior TEXT(30),
                stay TEXT(10),
                class INTEGER,
                predicted INTEGER
            );
        """
        load_query = f"LOAD CSV '{compas_dataset}' INTO MyCompas;"

        execute_query_fetch_all(self.evadb, create_table_query)
        execute_query_fetch_all(self.evadb, load_query)
        # print(execute_query_fetch_all(self.evadb, "SELECT * FROM MyCompas LIMIT 10;"))

        create_fn_query = (
            f"""CREATE FUNCTION IF NOT EXISTS DivExplorer
                IMPL  '{EvaDB_ROOT_DIR}/evadb/functions/divexplorer.py';"""
        )
        execute_query_fetch_all(self.evadb, create_fn_query)

        select_query = """
        SELECT DivExplorer(age, charge, race, sex, n_prior, stay, class, predicted) from MyCompas GROUP BY '6172 age';
        """
        # select_query = """
        # SELECT * FROM DivExplorer(MyCompas);
        # """
        print(execute_query_fetch_all(self.evadb, select_query))


    

