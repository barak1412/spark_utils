from pyspark.ml.stat import Summarizer
import pyspark.sql.functions as F
from pyspark.ml.linalg import DenseVector, VectorUDT


_abs_dense_vector_udf = F.udf(lambda v: DenseVector([abs(val) for val in v]), VectorUDT())


class DenseVectorVerticalNormalizer(object):
    def __init__(self, inputCol, outputCol, norm='l2'):
        self._norm = norm
        self._inputCol = inputCol
        self._outputCol = outputCol

    def transform(self, df):
        # verify we don't get empty dataframe
        first_row = df.select(self._inputCol).first()
        if first_row is None:
            return df.selectExpr('*', f'{self._inputCol} as {self._outputCol}')

        division_vector = self._get_division_vector(df)
        divide_dense_vector_udf = F.udf(lambda v: DenseVector([0 if v[index] == 0 else \
                                            v[index] / division_vector[index] for index in range(len(v))]), VectorUDT())
        return df.withColumn(self._outputCol, divide_dense_vector_udf(self._inputCol))

    def _get_division_vector(self, df):
        if self._norm == 'inf':
            # we should transform all vectors to positive values
            abs_df = df.withColumn(self._inputCol, _abs_dense_vector_udf(self._inputCol))
            result_vec = abs_df.agg(Summarizer.max(abs_df[self._inputCol]).alias(self._inputCol)).collect()[0][self._inputCol]
        else:
            agg_func = None
            if self._norm == 'l1':
                agg_func = Summarizer.normL1
            elif self._norm == 'l2':
                agg_func = Summarizer.normL2
            result_vec = df.agg(agg_func(df[self._inputCol]).alias(self._inputCol)).collect()[0][self._inputCol]

        return result_vec
