from dnasty.genetics import ConvGene
import unittest


def test_mutation(gene):
    for key in gene.exons.keys():
        gene.exons[key] += 1


class TestConvGene(unittest.TestCase):
    def setUp(self):
        self.exons = {"in_channels": 1, "out_channels": 2, "kernel_size": 3}
        self.loc = 1
        self.gene = ConvGene(in_channels=1, out_channels=2, kernel_size=3, loc=self.loc)

    def test_init(self):
        self.assertEqual(self.exons, self.gene.exons)
        self.assertEqual(self.loc, self.gene.location)

    def test_mutate(self):
        self.gene.mutate(test_mutation)
        self.assertEqual(self.gene.exons["in_channels"], self.exons["in_channels"] + 1)
        self.assertEqual(self.gene.exons["out_channels"], self.exons["out_channels"] + 1)
        self.assertEqual(self.gene.exons["kernel_size"], self.exons["kernel_size"] + 1)

    def test_len(self):
        self.assertEqual(len(self.gene), len(self.exons))


if __name__ == "__main__":
    unittest.main()