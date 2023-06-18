from spark_utils.preprocessing import DenseVectorVerticalNormalizer, SparseVectorVerticalNormalizer
import pyspark.ml.linalg as ml_linag
import pyspark.mllib.linalg as mllib_linag


class VerticalNormalizer(object):

    _SUPPORTED_NORMS = ['l1', 'l2', 'inf']

    def __init__(self, inputCol, outputCol, norm='l2'):
        if norm not in VerticalNormalizer._SUPPORTED_NORMS:
            raise Exception(f'unsupported norm `{norm}`, only {VerticalNormalizer._SUPPORTED_NORMS} are supported.')
        self._dense_vector_normalizer = DenseVectorVerticalNormalizer(inputCol=inputCol, outputCol=outputCol,
                                                                      norm=norm)
        self._sparse_vector_normalizer = SparseVectorVerticalNormalizer(inputCol=inputCol, outputCol=outputCol,
                                                                      norm=norm)
        self._inputCol = inputCol
        self._outputCol = outputCol

    def transform(self, df):
        # verify we don't get empty dataframe
        first_row = df.select(self._inputCol).first()
        if first_row is None:
            return df.selectExpr('*', f'{self._inputCol} as {self._outputCol}')
        first_row_vector = first_row[self._inputCol]
        if type(first_row_vector) in [mllib_linag.SparseVector, ml_linag.SparseVector]:
            return self._sparse_vector_normalizer.transform(df)
        elif type(first_row_vector) in [mllib_linag.DenseVector, ml_linag.DenseVector]:
            return self._dense_vector_normalizer.transform(df)
        else:
            raise Exception(f'invalid type {type(first_row_vector)}, only DenseVector or SparseVector are supported.')




