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
compas_trained_dataset = '/home/jeff/evadb/data/divexplorer/compas_test.csv'

class DivExplorerTest(unittest.TestCase):
    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        load_functions_for_testing(self.evadb, mode="debug")

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, "DROP TABLE IF EXISTS MyCompas;")
        execute_query_fetch_all(self.evadb, "DROP TABLE IF EXISTS MyCompasLearn;")
    
    def test_divexplorer_runs(self):
        from evadb.functions.divexplorer import DivExplorer

        divex = DivExplorer(
            min_support=0.1,
            ignore_cols=None,
            th_redundancy=None,
            top_k=None,
        )

        df = pd.read_csv(compas_dataset)
        result = divex(df)
        self.assertTrue(result is not None)

    def test_divexplorer_runs_with_top_k(self):
        from evadb.functions.divexplorer import DivExplorer

        divex = DivExplorer(
            min_support=0.05,
            ignore_cols=None,
            th_redundancy=0.05,
            top_k=10,
        )

        df = pd.read_csv(compas_dataset)
        result = divex(df)
        self.assertTrue(result is not None)

    def test_divexplorer_via_query(self):
        import time


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

        t = time.time()
        select_query = """
        SELECT DivExplorer(age, charge, race, sex, n_prior, stay, class, predicted) from MyCompas GROUP BY '6172 age';
        """
        # select_query = """
        # SELECT * FROM DivExplorer(MyCompas);
        # """
        result = execute_query_fetch_all(self.evadb, select_query)
        print(result._frames.sort_values(by=['divexplorer.d_fpr'], ascending=True))
        print('overall time', time.time() - t)

    def test_divexplorer_sklearn_train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv(compas_dataset)
        label_encoder_dicts = {}
        for col in df.columns:
            if col not in ['class', 'predicted']:
                l = LabelEncoder()
                df[col] = l.fit_transform(df[col])
                label_encoder_dicts[col] = l


        df_truth = df['class'].values.tolist().copy()
        df = df.drop(columns=['class', 'predicted'])
        df_X = df.values.tolist().copy()

        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_truth, test_size=0.20, random_state=random_state)
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        results = model.predict(X_test)
        print('accuracy', model.score(X_test, y_test))

        df_test = pd.DataFrame(X_test, columns=df.columns)
        df_test_actual = pd.DataFrame(y_test, columns=['class'])
        df_test_pred = pd.DataFrame(results, columns=['predicted'])
        df_test = pd.concat([df_test, df_test_actual, df_test_pred], axis=1)
        for col in df_test.columns:
            # Convert back to original values
            if col not in ['class', 'predicted']:
                df_test[col] = label_encoder_dicts[col].inverse_transform(df_test[col])

        df_test.to_csv(compas_trained_dataset, index=False)
        create_table_query = """
            CREATE TABLE IF NOT EXISTS MyCompasLearn (
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
        load_query = f"LOAD CSV '{compas_trained_dataset}' INTO MyCompasLearn;"
        execute_query_fetch_all(self.evadb, create_table_query)
        execute_query_fetch_all(self.evadb, load_query)

        create_fn_query = (
            f"""CREATE FUNCTION IF NOT EXISTS DivExplorer
                IMPL  '{EvaDB_ROOT_DIR}/evadb/functions/divexplorer.py';"""
        )
        execute_query_fetch_all(self.evadb, create_fn_query)

        select_query = """
        SELECT DivExplorer(age, charge, race, sex, n_prior, stay, class, predicted) from MyCompasLearn GROUP BY '1235 age';
        """
        # print(execute_query_fetch_all(self.evadb, select_query)._frames.sort_values(by=['d_fpr'], ascending=False))
        result = execute_query_fetch_all(self.evadb, select_query)
        print(result._frames.sort_values(by=['divexplorer.d_fpr'], ascending=True))