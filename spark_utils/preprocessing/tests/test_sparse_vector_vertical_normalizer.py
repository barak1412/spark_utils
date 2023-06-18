import unittest
import pandas as pd
import math
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.ml.linalg import SparseVector, VectorUDT
from spark_utils.preprocessing.sparse_vector_vertical_normalizer import SparseVectorVerticalNormalizer


class TestSparseVectorVerticalNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._spark = SparkSession.builder.master("local[1]") \
            .appName('test_app') \
            .getOrCreate()
        cls._input_col = 'VECTOR'
        cls._output_col = 'VECTOR_NORM'

    def test_empty_dataframe(self):
        schema = StructType([StructField(self._input_col, VectorUDT(), True)])
        empty_df = self._spark.createDataFrame([], schema=schema)
        normalizer = SparseVectorVerticalNormalizer(self._input_col, self._output_col)
        result_df = normalizer.transform(empty_df)
        self.assertTrue(self._input_col in result_df.columns)
        self.assertTrue(self._output_col in result_df.columns)
        self.assertEqual(result_df.count(), 0)
        self.assertEqual(len(result_df.columns), 2)

    def test_l1_norm(self):
        u1 = SparseVector(4, [1, 3], [3.0, 4.0])
        u2 = SparseVector(4, [0, 3], [-8.0, 3.0])
        u3 = SparseVector(4, [0], [2.0])

        u1_norm = SparseVector(4, [1, 3], [1.0, 4 / 7])
        u2_norm = SparseVector(4, [0, 3], [-0.8, 3 / 7])
        u3_norm = SparseVector(4, [0], [0.2])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = SparseVectorVerticalNormalizer(self._input_col, self._output_col, norm='l1')
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_l2_norm(self):
        u1 = SparseVector(5, [1, 3], [3.0, 4.0])
        u2 = SparseVector(5, [0, 3], [-8.0, 3.0])
        u3 = SparseVector(5, [0], [2.0])

        u1_norm = SparseVector(5, [1, 3], [1.0, 0.8])
        u2_norm = SparseVector(5, [0, 3], [-8/math.sqrt(68), 0.6])
        u3_norm = SparseVector(5, [0], [2/math.sqrt(68)])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = SparseVectorVerticalNormalizer(self._input_col, self._output_col, norm='l2')
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_inf_norm(self):
        u1 = SparseVector(5, [1, 3], [3.0, 4.0])
        u2 = SparseVector(5, [0, 3], [-8.0, 3.0])
        u3 = SparseVector(5, [0], [2.0])

        u1_norm = SparseVector(5, [1, 3], [1.0, 1.0])
        u2_norm = SparseVector(5, [0, 3], [-1.0, 0.75])
        u3_norm = SparseVector(5, [0], [0.25])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = SparseVectorVerticalNormalizer(self._input_col, self._output_col, norm='inf')
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_zero_sparse_vector(self):
        u1 = SparseVector(5, [1, 2, 3], [3.0, 0.0, 4.0])
        u2 = SparseVector(5, [0, 2, 3], [-8.0, 0.0, 3.0])

        u1_norm = SparseVector(5, [1, 3], [1.0, 0.8])
        u2_norm = SparseVector(5, [0, 3], [-1.0, 0.6])

        df = self._spark.createDataFrame([[u1], [u2]], [self._input_col])
        normalizer = SparseVectorVerticalNormalizer(self._input_col, self._output_col)
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_empty_sparse_vector(self):
        u1 = SparseVector(3, [], [])
        u2 = SparseVector(3, [], [])

        u1_norm = SparseVector(3, [], [])
        u2_norm = SparseVector(3, [], [])

        df = self._spark.createDataFrame([[u1], [u2]], [self._input_col])
        normalizer = SparseVectorVerticalNormalizer(self._input_col, self._output_col)
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

