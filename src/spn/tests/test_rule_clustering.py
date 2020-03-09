from spn.structure.Base import Rule, Condition
import numpy as np
import unittest

class TestRuleStructures(unittest.TestCase):
    def test_basic_Rule(self):
        c1 = Condition(0, np.less, 7)
        c2 = Condition(1, np.equal, 100)
        r1 = Rule([c1, c2])

        c3 = Condition(0, np.less, 5)
        c4 = Condition(1, np.equal, 100)
        c5 = Condition(0, np.greater_equal, -10.5)
        r2 = Rule([c3, c4, c1, c5])

        m = r1.merge(r2)
        assert len(m) == 4
        x1 = [-10.5,100]
        assert m.apply(x1)
        x2 = [-50, 2000]
        assert not m.apply(x2)
        x3 = [-10.5, 0]
        assert not m.apply(x3)

if __name__ == '__main__':
    unittest.main()
