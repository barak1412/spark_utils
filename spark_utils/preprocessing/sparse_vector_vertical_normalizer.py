from pyspark.ml.linalg import SparseVector, VectorUDT
import pyspark.sql.functions as F
import math


class SparseVectorVerticalNormalizer(object):
    def __init__(self, inputCol, outputCol, norm='l2'):
        self._norm = norm
        self._inputCol = inputCol
        self._outputCol = outputCol

    def transform(self, df):
        # verify we don't get empty dataframe
        first_row = df.select(self._inputCol).first()
        if first_row is None:
            return df.selectExpr('*', f'{self._inputCol} as {self._outputCol}')

        # calculate the aggregated devided vector
        first_vector = first_row[self._inputCol]
        devided_sparse_vector = self._get_division_vector(df, first_vector.size)

        # normalize
        _divide_sparse_vector_udf = F.udf(lambda v: SparseVector(v.size, [index for index in v.indices if v[int(index)] != 0],
                                                                [val / devided_sparse_vector[int(index)] for
                                                                (index, val) in zip(v.indices, v.values) if val != 0]),
                                          VectorUDT())
        return df.withColumn(self._outputCol, _divide_sparse_vector_udf(self._inputCol))

    def _get_division_vector(self, df, output_vector_dim):
        if self._norm == 'inf':
            map_func = lambda v: abs(v)
            agg_func = lambda a, b: max(a, b)
            post_agg_func = lambda v: v
        elif self._norm == 'l1':
            map_func = lambda v: abs(v)
            agg_func = lambda a, b: a + b
            post_agg_func = lambda v: v
        elif self._norm == 'l2':
            map_func = lambda v: v ** 2
            agg_func = lambda a, b: a + b
            post_agg_func = lambda v: math.sqrt(v)
        result_vec_tupled = df.rdd.flatMap(lambda row: [t for t in zip(row[self._inputCol].indices,
                                                            [map_func(v) for v in
                                                            row[self._inputCol].values]) if t[1] != 0]) \
            .reduceByKey(agg_func) \
            .collect()
        result_vec_tupled.sort(key=lambda tup: tup[0])
        result_vec = SparseVector(output_vector_dim, [index_value_pair[0] for index_value_pair in result_vec_tupled],
                                  [post_agg_func(index_value_pair[1]) for index_value_pair in result_vec_tupled])

        return result_vec