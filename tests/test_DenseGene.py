from dnasty.genetics import LinearGene
import unittest


class TestLinearGene(unittest.TestCase):
    def setUp(self):
        self.exons = {"in_features": 10, "out_features": 110, "dropout": True}
        self.gene = LinearGene(in_features=10, out_features=110, dropout=True)

    def test_init(self):
        self.assertEqual(self.gene.exons, self.exons)

        false_in_features = {"in_features": 1, "out_features": 100, "dropout": True}
        self.assertRaises(ValueError, LinearGene, **false_in_features)

        false_out_features = {"in_features": 9, "out_features": 100_000, "dropout": True}
        self.assertRaises(ValueError, LinearGene, **false_out_features)

        false_dropout = {"in_features": 9, "out_features": 100, "dropout": "hi"}
        self.assertRaises(ValueError, LinearGene, **false_dropout)

    def test_mutate(self):
        self.gene.mutate()
        self.assertIn(self.gene.exons["out_features"], LinearGene.allowed_features)
        self.assertTrue(self.gene.exons["in_features"] >= LinearGene.allowed_features[0])

    def test_len(self):
        self.assertEqual(len(self.gene), 3)

    def test_validate(self):
        pass

    def test_getattr(self):
        self.assertEqual(self.gene.in_features, 10)
        self.assertEqual(self.gene.out_features, 110)
        self.assertEqual(self.gene.dropout, True)
        self.assertEqual(self.gene.allowed_features, range(9, 10_000))


if __name__ == "__main__":
    unittest.main()
