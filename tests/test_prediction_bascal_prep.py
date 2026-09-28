import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orca.scripts.prediction_bascal_prep import index_pileup


class TestIndexPileup(unittest.TestCase):
    def test_indexes_final_contig_and_byte_ranges(self):
        cases = [
            ([], []),
            (["only", "only"], ["only"]),
            (["first", "first", "last", "last"], ["first", "last"]),
        ]
        for contigs, expected_ids in cases:
            with self.subTest(contigs=contigs), TemporaryDirectory() as directory:
                pileup = Path(directory) / "input.pileup"
                index = Path(directory) / "input.pileup.index"
                lines = [
                    f"{contig}\t1\tA\t10\t..........\tIIIIIIIIII\n"
                    for contig in contigs
                ]
                pileup.write_text("".join(lines), encoding="ascii")

                index_pileup(pileup, index)

                with index.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["id"] for row in rows], expected_ids)

                data = pileup.read_bytes()
                for row in rows:
                    start, end = int(row["start"]), int(row["end"])
                    self.assertLess(start, end)
                    self.assertLessEqual(end, len(data))
                    self.assertEqual(
                        {line.split(b"\t", 1)[0] for line in data[start:end].splitlines()},
                        {row["id"].encode("ascii")},
                    )
                self.assertEqual(
                    [(int(row["start"]), int(row["end"])) for row in rows],
                    [
                        (sum(len(line.encode("ascii")) for line in lines[:contigs.index(contig)]),
                         sum(len(line.encode("ascii")) for line in lines[:(
                             contigs.index(expected_ids[i + 1])
                             if i + 1 < len(expected_ids) else len(lines)
                         )]))
                        for i, contig in enumerate(expected_ids)
                    ],
                )


if __name__ == "__main__":
    unittest.main()
