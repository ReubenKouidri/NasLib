from dnasty.genetics import Genome, ConvBlock2dGene, MaxPool2dGene
import coverage
import unittest


class TestGenome(unittest.TestCase):

    def setUp(self):
        self.conv = ConvBlock2dGene(1, 32, 10)
        self.conv2 = ConvBlock2dGene(1, 50, 10)
        self.mp = MaxPool2dGene(2, 2)
        self.genes = [self.conv, self.conv2, self.mp]
        self.genome = Genome(self.genes)

    def test_init(self):
        self.assertEqual(self.conv, self.genome.genes[0])
        self.assertEqual(self.conv2, self.genome.genes[1])
        self.assertEqual(self.mp, self.genome.genes[2])
        self.assertEqual(self.genome.fitness, 0.0)
        self.assertIsInstance(self.genome.genes, list)

    def test_crossover(self):
        pass

    def test_len(self):
        self.assertEqual(len(self.genome), 3)

    def test_sync(self):
        self.assertNotEqual(self.conv.out_channels, self.conv2.in_channels)
        self.genome._sync_layers()
        self.assertEqual(self.conv.out_channels, self.conv2.in_channels)

    def test_outdims(self):
        self.assertEqual(self.genome.outdims, 55)


if __name__ == "__main__":
    cov = coverage.Coverage()
    cov.start()
    unittest.main()
    cov.stop()
    cov.save()
    cov.report()
