from dnasty.genetics import DenseGene
import unittest


def test_mutation(gene):
    for key in gene.exons.keys():
        gene.exons[key] += 1


class TestDenseGene(unittest.TestCase):
    def setUp(self):
        self.exons = {"in_features": 1, "out_features": 2}
        self.loc1 = 1
        self.loc2= 2
        self.gene1 = DenseGene(in_features=1, out_features=2, loc=self.loc1, dropout=True)
        self.gene2 = DenseGene(in_features=1, out_features=4, loc=self.loc2, dropout=True)

    def test_init(self):
        self.assertEqual(self.exons, self.gene1.exons)
        self.assertEqual(self.loc1, self.gene1.location)

    def test_mutate(self):
        in_features = 1
        out_features = 2
        gene = DenseGene(in_features, out_features, loc=1)
        gene.mutate(test_mutation)
        self.assertEqual(gene.exons["in_features"], 1)
        self.assertEqual(gene.exons["out_features"], 3)
        self.assertEqual(gene.location, 1)

    def test_len(self):
        self.assertEqual(len(self.gene1), 2)


if __name__ == "__main__":
    unittest.main()
