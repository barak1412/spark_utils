import unittest
import pandas as pd
import math
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.ml.linalg import VectorUDT
import pyspark.ml.linalg as ml_linag
import pyspark.mllib.linalg as mllib_linag
from spark_utils.preprocessing.vertical_normalizer import VerticalNormalizer


class TestVerticalNormalizer(unittest.TestCase):
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
        normalizer = VerticalNormalizer(self._input_col, self._output_col)
        result_df = normalizer.transform(empty_df)
        self.assertTrue(self._input_col in result_df.columns)
        self.assertTrue(self._output_col in result_df.columns)
        self.assertEqual(result_df.count(), 0)
        self.assertEqual(len(result_df.columns), 2)

    def test_mllib_dense_norm(self):
        u1 = mllib_linag.DenseVector([0.0, 3.0, 0.0, 4.0])
        u2 = mllib_linag.DenseVector([-8.0, 0.0, 0.0, 3.0])
        u3 = mllib_linag.DenseVector([2.0, 0.0, 0.0, 0.0])

        u1_norm = ml_linag.DenseVector([0.0, 1.0, 0.0, 0.8])
        u2_norm = ml_linag.DenseVector([-8 / math.sqrt(68), 0.0, 0.0, 0.6])
        u3_norm = ml_linag.DenseVector([2 / math.sqrt(68), 0.0, 0.0, 0.0])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = VerticalNormalizer(self._input_col, self._output_col)
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_ml_dense_norm(self):
        u1 = ml_linag.DenseVector([0.0, 3.0, 0.0, 4.0])
        u2 = ml_linag.DenseVector([-8.0, 0.0, 0.0, 3.0])
        u3 = ml_linag.DenseVector([2.0, 0.0, 0.0, 0.0])

        u1_norm = ml_linag.DenseVector([0.0, 1.0, 0.0, 0.8])
        u2_norm = ml_linag.DenseVector([-8 / math.sqrt(68), 0.0, 0.0, 0.6])
        u3_norm = ml_linag.DenseVector([2 / math.sqrt(68), 0.0, 0.0, 0.0])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = VerticalNormalizer(self._input_col, self._output_col)
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1, u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_mllib_sparse_norm(self):
        u1 = mllib_linag.SparseVector(5, [1, 3], [3.0, 4.0])
        u2 = mllib_linag.SparseVector(5, [0, 3], [-8.0, 3.0])
        u3 = mllib_linag.SparseVector(5, [0], [2.0])

        u1_norm = ml_linag.SparseVector(5, [1, 3], [1.0, 0.8])
        u2_norm = ml_linag.SparseVector(5, [0, 3], [-8/math.sqrt(68), 0.6])
        u3_norm = ml_linag.SparseVector(5, [0], [2/math.sqrt(68)])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = VerticalNormalizer(self._input_col, self._output_col)
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)

    def test_ml_sparse_norm(self):
        u1 = ml_linag.SparseVector(5, [1, 3], [3.0, 4.0])
        u2 = ml_linag.SparseVector(5, [0, 3], [-8.0, 3.0])
        u3 = ml_linag.SparseVector(5, [0], [2.0])

        u1_norm = ml_linag.SparseVector(5, [1, 3], [1.0, 0.8])
        u2_norm = ml_linag.SparseVector(5, [0, 3], [-8/math.sqrt(68), 0.6])
        u3_norm = ml_linag.SparseVector(5, [0], [2/math.sqrt(68)])

        df = self._spark.createDataFrame([[u1], [u2], [u3]], [self._input_col])
        normalizer = VerticalNormalizer(self._input_col, self._output_col)
        result_pd = normalizer.transform(df).toPandas()
        expected_result_pd = pd.DataFrame(data=[[u1,u1_norm], [u2, u2_norm], [u3, u3_norm]],
                                          columns=[self._input_col, self._output_col])
        pd.testing.assert_frame_equal(result_pd, expected_result_pd)
