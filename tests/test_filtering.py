import pytest
import numpy as np
from pyriemann_qiskit.utils.filtering import NoDimRed, NaiveDimRed


class TestCommon:
    @pytest.mark.parametrize(
        "dim_red", [NoDimRed(), NaiveDimRed(), NaiveDimRed(is_even=False)]
    )
    def test_fit(self, dim_red):
        """Ensure all filtering can be instanciated,
        and fitted.
        """
        X = np.array([[]])
        y = None
        assert dim_red.fit(X, y) == dim_red

    @pytest.mark.parametrize(
        "dim_red", [NoDimRed(), NaiveDimRed(), NaiveDimRed(is_even=False)]
    )
    def test_transform_empty_array(self, dim_red):
        """Ensure we can provide an empty array to the
        transform method.
        """
        X = np.array([[]])
        assert dim_red.transform(X).shape == (1, 0)


class TestNaiveDimRed:
    @pytest.mark.parametrize(
        "dim_red", [(NaiveDimRed(), True), (NaiveDimRed(is_even=False), False)]
    )
    def test_reduction(self, dim_red):
        """Ensure the dimension of the feature is divied by two
        and the the adequate indices are kept depending on wether the
        transform is Even or Odd.
        """
        X = np.array([[True, False], [True, False]])
        instance = dim_red[0]
        isEven = dim_red[1]
        X2 = instance.transform(X)
        assert X2.shape == (2, 1)
        assert X2[0][0] == isEven
        assert X2[1][0] == isEven
