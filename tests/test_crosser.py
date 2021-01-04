from tests.utils_testing import *


# ==============================================================================
# Crosser
# ==============================================================================

# Base crosser
@pytest.fixture
def crosser():
    return Crosser(0.6, True)


@pytest.fixture
def trees():
    trees = []
    for i in range(3):
        trees.append(build_trees(5, 1)[0])
    return trees


# +++++++++++++++
# Init
# +++++++++++++++

@pytest.mark.parametrize("prob, is_both",
                         [(0.1, True),
                          (0.5, False)])
def test_crosser_init(prob, is_both):
    crosser = Crosser(prob, is_both)
    assert prob == crosser.cross_prob
    assert is_both == crosser.cross_both

# +++++++++++++++
# Set params
# +++++++++++++++

@pytest.mark.parametrize("cross_prob", [0.1, 0.2, 0.4])
def test_set_cross_prob(crosser, cross_prob):
    crosser.set_params(cross_prob=cross_prob)
    assert crosser.cross_prob == cross_prob


@pytest.mark.parametrize("cross_prob", [-1, -0.1, 0])
def test_set_cross_prob_below_0(crosser, cross_prob):
    crosser.set_params(cross_prob=cross_prob)
    assert crosser.cross_prob == 0


@pytest.mark.parametrize("cross_prob", [1, 1.1, 10])
def test_set_cross_prob_above_1(crosser, cross_prob):
    crosser.set_params(cross_prob=cross_prob)
    assert crosser.cross_prob == 1


@pytest.mark.parametrize("cross_prob", ["string", [1]])
def test_set_cross_prob_wrong_type(crosser, cross_prob):
    with pytest.raises(TypeError):
        crosser.set_params(cross_prob=cross_prob)


@pytest.mark.parametrize("is_both", [True, False])
def test_set_is_both(crosser, is_both):
    crosser.set_params(cross_both=is_both)
    assert crosser.cross_both == is_both


@pytest.mark.parametrize("is_both", ["string", [True]])
def test_set_is_both_wrong_type(crosser, is_both):
    with pytest.raises(TypeError):
        crosser.set_params(cross_both=is_both)


# ==============================================================================
# Cross low level
# ==============================================================================
